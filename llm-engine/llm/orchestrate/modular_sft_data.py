"""Authorized parent-only SFT rendering and bounded probe preparation without model or tensor loading."""

from dataclasses import dataclass
from collections.abc import Callable
import json
import logging
from pathlib import Path
from threading import Event

import numpy as np
from tokenizers import Tokenizer

from llm import storage
from llm.architecture import package
from llm.architecture.package_integrity import METADATA_BYTES, read_bytes
from llm.architecture.spec import canonical_digest
from llm.architecture.tokenizer_binding import TOKENIZER_BYTES
from llm.chat import template
from llm.chat.template import Reasoning, Turn
from llm.config import SftConfig
from llm.corpus import pack, remote, sft
from llm.orchestrate.modular_data import PreparedData
from llm.runtime.artifacts import ArtifactStore
from llm.runtime.protocol import RuntimeArtifact
from llm.train.diagnostics import DIAGNOSTIC_ROWS
from llm.train import mapped_data
from llm.train.modular_records import SftDataset, SftInputs, TrainingPublication
from llm.train.plan import DataPlan
from llm.train.progress import BuildReport, ProgressCallback
from llm.train.sft_tokenizer import prepare_tokenizer

PROBE_BYTES = 8 << 20
PROBE_LINE_BYTES = 256 << 10
PROBE_FILES = 128
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedSftData:
    """Own private aligned inputs/provenance until the existing handler grants them to its child."""

    data: PreparedData
    mask: Path
    dataset: SftDataset
    pretrain_probe: Path | None = None
    chat_probe: Path | None = None

    @property
    def additional_paths(self) -> tuple[Path, ...]:
        """Return every extra actual file that parent admission must charge before session staging."""
        return tuple(path for path in (self.mask, self.pretrain_probe, self.chat_probe) if path is not None)

    def grant_inputs(self, artifacts: ArtifactStore, cfg: SftConfig, reference: RuntimeArtifact | None) -> SftInputs:
        """Stage finite mask/probe inputs with distinct formats and the original public SFT request."""
        return SftInputs(request=cfg, dataset=self.dataset, mask=artifacts.stage_file(self.mask, "loss_mask_v1"),
            pretrain_probe=artifacts.stage_file(self.pretrain_probe, "sft_probe_v1") if self.pretrain_probe else None,
            chat_probe=artifacts.stage_file(self.chat_probe, "sft_probe_v1") if self.chat_probe else None, reference=reference)


def pretrain_probe(base: package.VerifiedPackage, context_window: int, vocabulary: int,
                   check_cancelled: Callable[[], None]) -> np.ndarray | None:
    """Sample finite evenly spaced windows of the base's own derived corpus, never scanning or copying the complete mapping."""
    metadata = TrainingPublication.model_validate_json(read_bytes(base.directory / "meta.json", METADATA_BYTES))
    key = metadata.result.get("corpus_key")
    if not isinstance(key, str):
        return None
    if not key or len(key) > 255 or Path(key).name != key or key in {".", ".."} or "\x00" in key:
        raise ValueError("SFT base pretraining probe has an invalid owned corpus key")
    path = storage.pretrain_corpus_path(key)
    width = context_window + 1
    if not path.is_file() or DIAGNOSTIC_ROWS * width * 8 > PROBE_BYTES:
        return None
    corpus = np.memmap(path, dtype=np.uint16, mode="r")
    if len(corpus) < DIAGNOSTIC_ROWS * width:
        return None
    stride = (len(corpus) - width) // max(1, DIAGNOSTIC_ROWS - 1)
    rows = []
    for index in range(DIAGNOSTIC_ROWS):
        check_cancelled()
        rows.append(np.asarray(corpus[index * stride:index * stride + width], dtype=np.int64))
        mapped_data.release_pages(corpus)
    probe = np.stack(rows)
    return probe if np.all(probe < vocabulary) else None


def probe_turns(conversation: object) -> tuple[str | None, list[Turn]] | None:
    """Validate bounded optional held-out JSON before canonical rendering; malformed rows supply no measurement."""
    if not isinstance(conversation, dict):
        return None
    system, turns = conversation.get("system"), conversation.get("turns")
    if system is not None and not isinstance(system, str):
        return None
    if not isinstance(turns, list) or not 1 <= len(turns) <= 512:
        return None
    for turn in turns:
        if not isinstance(turn, dict) or not isinstance(turn.get("role", ""), str) or not isinstance(turn.get("text", ""), str):
            return None
        if any(turn.get(key) is not None and not isinstance(turn[key], str) for key in ("tool", "argument")):
            return None
    parsed = sft._turns(conversation)
    return (system, parsed) if parsed is not None else None


def chat_probe(tokenizer: Tokenizer, context_window: int, reasoning: Reasoning,
               check_cancelled: Callable[[], None]) -> np.ndarray | None:
    """Stream a bounded held-out JSONL prefix and stop after enough canonical SFT tokens; oversized optional probes are omitted."""
    directory = storage.eval_dir() / "chat"
    if not directory.is_dir() or DIAGNOSTIC_ROWS * context_window * 8 > PROBE_BYTES:
        return None
    paths = []
    for path in directory.glob("*.jsonl"):
        paths.append(path)
        if len(paths) > PROBE_FILES:
            LOG.warning("SFT held-out chat exceeds its file budget; KL measurement omitted")
            return None
    ids, consumed = [], 0
    wanted = DIAGNOSTIC_ROWS * context_window
    for path in sorted(paths):
        with path.open("rb") as source:
            while len(ids) < wanted:
                check_cancelled()
                line = source.readline(min(PROBE_LINE_BYTES, PROBE_BYTES - consumed) + 1)
                if not line:
                    break
                consumed += len(line)
                if consumed > PROBE_BYTES or len(line) > PROBE_LINE_BYTES:
                    LOG.warning("SFT held-out chat exceeds its byte/line budget; KL measurement omitted")
                    return None
                try:
                    conversation = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                parsed = probe_turns(conversation)
                if parsed is not None:
                    system, turns = parsed
                    rendered, _ = template.build_training_example(system, turns, tokenizer, None, reasoning)
                    ids.extend(rendered[:wanted - len(ids)])
        if len(ids) >= wanted:
            break
    rows = min(DIAGNOSTIC_ROWS, len(ids) // context_window)
    return np.asarray(ids[:rows * context_window], dtype=np.int64).reshape(rows, context_window) if rows else None


def prepare(cfg: SftConfig, *, saved: package.VerifiedPackage, base: package.VerifiedPackage,
            directory: Path, actor: str | None, progress: ProgressCallback, cancel: Event,
            budget: pack.PackBudget) -> PreparedSftData:
    """Render authorized conversations privately, preserve exact masks and derive tokenizer identity before child admission."""
    def check_cancelled() -> None:
        """Stop acquisition, bounded probes and mapped identity scans under the actual parent request's cancellation owner."""
        if cancel.is_set():
            raise InterruptedError("SFT cancelled during parent data preparation")

    def report_progress(frame) -> None:
        """Observe cancellation before forwarding actual rendering work to the durable job."""
        check_cancelled()
        progress(frame)

    raw = read_bytes(saved.directory / "tokenizer.json", TOKENIZER_BYTES)
    raw, binding = prepare_tokenizer(raw, saved.manifest.definition.tokenizer, [])
    window = saved.manifest.definition.context_window
    tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
    reasoning = Reasoning.STRIP if cfg.strip_thinking else Reasoning.KEEP
    report = BuildReport(report_progress)
    dataset_dir = directory / "dataset"
    with remote.Client(actor) as client:
        selection = sft.resolve(client, cfg.datasets)
        report.at("rendering", documents=sum(value.taken for value in selection) or None, documents_done=0)
        stats = sft.build(client, tokenizer, selection, reasoning, report, directory=dataset_dir,
                          budget=budget, check_cancelled=check_cancelled)
    check_cancelled()
    raw, binding = prepare_tokenizer(raw, binding, stats.special_ids_seen)
    tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
    key = "sft-" + canonical_digest({"sets": [value.identity() for value in selection],
        "reasoning": reasoning.value, "tokenizer": binding.fingerprint})
    corpus, mask = sft.load(key, directory=dataset_dir)
    eot, documents, info = sft.load_segments(key, directory=dataset_dir)
    plan = DataPlan(documents=documents, eot_id=eot, shuffle=cfg.shuffle, seed=cfg.seed)
    tokenizer_path = directory / "tokenizer.json"
    tokenizer_path.write_bytes(raw)
    counts = tuple(info["token_counts"]) if info.get("token_counts") is not None else None
    data = PreparedData(storage.sft_corpus_path(key, directory=dataset_dir), storage.sft_spans_path(key, directory=dataset_dir),
        tokenizer_path, binding, len(corpus), tuple(documents.names), (), eot, key,
        mapped_data.fingerprint(corpus, mask, plan=plan, check_cancelled=check_cancelled), counts, cfg.shuffle, cfg.seed)
    provenance = SftDataset(sets=tuple(value.record() for value in selection), reasoning=reasoning.value,
        conversations=stats.conversations, total_tokens=stats.total_tokens, trainable_tokens=stats.trainable_tokens,
        skipped=stats.skipped, special_ids_seen=tuple(stats.special_ids_seen))
    probes = {}
    for name, rows in (("pretrain_probe", pretrain_probe(base, window, binding.vocabulary_size, check_cancelled)),
                       ("chat_probe", chat_probe(tokenizer, window, reasoning, check_cancelled) if cfg.measure_base_kl else None)):
        if rows is not None:
            path = directory / f"{name}.npy"
            np.save(path, rows, allow_pickle=False)
            probes[name] = path
    report.finish(conversations=stats.conversations, total_tokens=stats.total_tokens,
                  trainable_tokens=stats.trainable_tokens, skipped_too_long=stats.skipped)
    return PreparedSftData(data, storage.sft_mask_path(key, directory=dataset_dir), provenance, **probes)
