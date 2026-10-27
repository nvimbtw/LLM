"""Parent-only authorized corpus/tokenizer preparation; no source import or model/tensor loading."""

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from uuid import UUID

from tokenizers import Tokenizer

from llm import storage
from llm.architecture import package
from llm.architecture.package_integrity import METADATA_BYTES, decode_json, read_bytes
from llm.architecture.spec import FinalTokenizer
from llm.architecture.tokenizer_binding import TOKENIZER_BYTES, binding_from_bytes
from llm.config import TrainConfig
from llm.corpus import pack, remote, spans as span_table, vocab
from llm.orchestrate import train as legacy
from llm.runtime.artifacts import ArtifactStore
from llm.train.modular_records import MAX_TRAINING_DOCUMENTS, CorpusPlan, DataStage
from llm.train.plan import DataPlan, Stage
from llm.train.progress import BuildReport, ProgressCallback
from llm.train import mapped_data
from llm.train.sft_tokenizer import marker_history

MAX_CORPUS_BYTES = 16 << 30
MAX_DOCUMENTS = MAX_TRAINING_DOCUMENTS


@dataclass(frozen=True)
class PreparedData:
    """Own parent-verified local inputs until the session snapshots them into private grants."""

    corpus: Path
    spans: Path
    tokenizer: Path
    binding: FinalTokenizer
    tokens: int
    names: tuple[str, ...]
    stages: tuple[DataStage, ...]
    eot_id: int
    key: str
    fingerprint: str
    token_counts: tuple[int, ...] | None
    shuffle: bool
    seed: int
    special_ids_seen: tuple[int, ...] | None = None

    def grant_plan(self, artifacts: ArtifactStore) -> CorpusPlan:
        """Materialize the exact parent span table and expose bounded, path-free data-plan facts."""
        return CorpusPlan(tokens=self.tokens, spans=artifacts.stage_file(self.spans, "span_table_v1"),
            names=self.names, stages=self.stages, eot_id=self.eot_id, corpus_key=self.key,
            fingerprint=self.fingerprint, token_counts=self.token_counts, shuffle=self.shuffle, seed=self.seed,
            special_ids_seen=self.special_ids_seen)


def data_plan(documents: span_table.Spans, eot_id: int, cfg: TrainConfig, stage_metadata: list[dict], tokens: int) -> DataPlan:
    """Preserve packed stage membership while taking current names/epoch counts from the request."""
    if not documents:
        return DataPlan.whole_corpus(tokens)
    requested = cfg.stages()
    stages = [Stage(name=requested[index].name if index < len(requested) else value.get("name"),
                    epochs=requested[index].epochs if index < len(requested) else value.get("epochs", cfg.epochs),
                    documents=documents.in_stage(index)) for index, value in enumerate(stage_metadata)]
    return DataPlan(documents=documents, eot_id=eot_id, shuffle=cfg.shuffle, seed=cfg.seed, stages=stages)


def tokenizer_bytes(cfg: TrainConfig, client: remote.Client | None, selection: list[pack.StageSelection], report: BuildReport | None,
                    names: tuple[str, ...], saved: package.VerifiedPackage | None) -> tuple[bytes, FinalTokenizer]:
    """Use the saved package's private vocabulary on resume, or acquire the requested/derived owned build."""
    if saved is not None:
        binding = saved.manifest.definition.tokenizer
        if cfg.tokenizer_id is not None and cfg.tokenizer_id != str(binding.tokenizer_id):
            raise ValueError("resume tokenizer differs from the saved definition")
        raw = read_bytes(saved.directory / "tokenizer.json", TOKENIZER_BYTES)
        actual = binding_from_bytes(binding.tokenizer_id, raw, tuple(binding.reserved_token_ids))
        if actual.model_copy(update={"derived_from": binding.derived_from}) != binding:
            raise ValueError("saved tokenizer facts differ from its owned bytes")
        return raw, binding
    if client is None or report is None:
        raise ValueError("new vocabulary acquisition requires an authorized parent data client and reporter")
    tokenizer_id = legacy._vocabulary(cfg, client, selection, report)
    raw = read_bytes(storage.tokenizer_file(storage.tokenizer_build_dir(tokenizer_id)), TOKENIZER_BYTES)
    binding = binding_from_bytes(UUID(tokenizer_id), raw, names)
    if vocab.fingerprint(tokenizer_id) != binding.fingerprint[:vocab.FINGERPRINT_CHARS]:
        raise ValueError("tokenizer build fingerprint differs from owned bytes")
    return raw, binding


def prepare(cfg: TrainConfig, *, context_window: int, reserved_names: tuple[str, ...], directory: Path,
            actor: str | None, progress: ProgressCallback, cancel: Event,
            saved: package.VerifiedPackage | None = None, budget: pack.PackBudget | None = None) -> PreparedData:
    """Resolve authorized datasets, build/reuse packed data and copy tokenizer/spans without constructing a model."""
    effective = cfg.model_copy(update={"ctx_window": context_window})
    budget = budget or pack.PackBudget(max_corpus_bytes=MAX_CORPUS_BYTES, max_documents=MAX_DOCUMENTS)

    def check_cancelled() -> None:
        """Stop bounded packing/identity work at a caller-owned cooperative cancellation boundary."""
        if cancel.is_set():
            raise InterruptedError("training cancelled during data preparation")

    def report_progress(frame) -> None:
        """Observe cooperative data-phase cancellation before forwarding measured build progress."""
        check_cancelled()
        progress(frame)
    report = BuildReport(report_progress)
    with remote.Client(actor) as client:
        selection = pack.resolve(client, cfg.stages())
        raw, binding = tokenizer_bytes(cfg, client, selection, report, reserved_names, saved)
        if cancel.is_set():
            raise InterruptedError("training cancelled during data preparation")
        tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
        short_hash = binding.fingerprint[:vocab.FINGERPRINT_CHARS]
        key = pack.corpus_key(short_hash, selection)
        dataset = None
        if cfg.rebuild_dataset or not pack.is_built(key, selection, short_hash):
            dataset = legacy.build_dataset(effective, client, tokenizer, key=key, selection=selection,
                tokenizer_id=str(binding.tokenizer_id), fingerprint=short_hash, report=report,
                budget=budget)
    path = storage.pretrain_corpus_path(key)
    if not 0 < path.stat().st_size <= budget.max_corpus_bytes:
        raise ValueError("packed corpus exceeds training input byte budget or is empty")
    corpus = pack.load_corpus(key)
    eot_id, documents, info = pack.load_segments(key)
    if len(documents) > budget.max_documents:
        raise ValueError("packed corpus exceeds training document row budget")
    legacy._phase_one(effective, tokenizer, dataset, corpus, documents, info, report)
    plan = data_plan(documents, eot_id, effective, info.get("stages", []), len(corpus))
    directory.mkdir(parents=True, exist_ok=True)
    spans_path, tokenizer_path = directory / "spans.npy", directory / "tokenizer.json"
    plan.documents.save(spans_path)
    tokenizer_path.write_bytes(raw)
    stages = tuple(DataStage(name=stage.name, epochs=stage.epochs) for stage in plan.stages)
    counts = tuple(info["token_counts"]) if info.get("token_counts") is not None else None
    seen = marker_history({"result": {"dataset": info}}, binding.vocabulary_size)
    if saved is not None:
        previous = marker_history(decode_json(read_bytes(saved.directory / "meta.json", METADATA_BYTES)), binding.vocabulary_size)
        seen = sorted(set(seen) | set(previous)) if seen is not None and previous is not None else None
    return PreparedData(path, spans_path, tokenizer_path, binding, len(corpus), tuple(plan.documents.names),
                        stages, plan.eot_id, key, mapped_data.fingerprint(corpus, plan=plan, check_cancelled=check_cancelled),
                        counts, plan.shuffle, plan.seed, tuple(seen) if seen is not None else None)
