"""Worker-owned inherited SFT weights, token initialization and bounded numeric probe inputs."""

from dataclasses import asdict, replace
from collections.abc import Callable

import numpy as np
from pydantic import TypeAdapter

from llm.architecture import package, package_transfer
from llm.architecture.build import RecipeModule
from llm.architecture.interfaces import TokenRowResult
from llm.architecture.package_execution import validate_saved_tensors
from llm.architecture.package_integrity import METADATA_BYTES, encode_json, read_bytes
from llm.architecture.spec import ResolvedDefinition
from llm.architecture.tokenizer_binding import TOKENIZER_BYTES
from llm.model import checkpoint
from llm.runtime.protocol import RuntimeArtifact, Train
from llm.runtime.worker import WorkerContext
from llm.train.diagnostics import DIAGNOSTIC_ROWS, Probes
from llm.train.modular_records import TrainingPreparation, TrainingPublication
from llm.train.model_policy import optimizer_groups
from llm.train.sft_tokenizer import marker_history, prepare_tokenizer
from llm.train.warm_start import initialize_token_rows, untrained_marker_ids

ROW_RESULT = TypeAdapter(TokenRowResult)


def require_definition(checked: package.VerifiedPackage, definition: ResolvedDefinition,
                       tokenizer: bytes, used_ids: list[int]) -> None:
    """Require the exact saved structure plus only a byte-verified compatible memory-label derivation."""
    source = checked.manifest.definition
    raw = read_bytes(checked.directory / "tokenizer.json", TOKENIZER_BYTES)
    expected, binding = prepare_tokenizer(raw, source.tokenizer, used_ids)
    if (definition != source.model_copy(update={"tokenizer": binding}) or tokenizer != expected
            or checked.manifest.parameter_precision not in (None, "fp32")):
        raise ValueError("SFT definition differs from its saved structure or compatible tokenizer bytes")


def load_probe(grant: RuntimeArtifact | None, *, columns: int, vocabulary_size: int,
               context: WorkerContext) -> np.ndarray | None:
    """Read one small readonly numeric NPY probe without allowing archive decompression or object arrays."""
    if grant is None:
        return None
    if grant.size_bytes > DIAGNOSTIC_ROWS * columns * 8 + 1024:
        raise ValueError("SFT probe exceeds its row/context byte budget")
    path = context.artifacts.verify(grant)
    with path.open("rb") as source:
        if source.read(6) != b"\x93NUMPY":
            raise ValueError("SFT probe must be a single numeric NPY array")
    rows = np.load(path, mmap_mode="r", allow_pickle=False)
    if (rows.dtype != np.int64 or rows.ndim != 2 or not 1 <= rows.shape[0] <= DIAGNOSTIC_ROWS
            or rows.shape[1] != columns or np.any(rows < 0) or np.any(rows >= vocabulary_size)):
        raise ValueError("SFT probe shape, dtype or token IDs differ from the inherited model")
    context.check_cancelled()
    return rows


class SftOperations:
    """Own verified input packages and SFT-specific restore/initialization callbacks through one worker request."""

    def __init__(self, command: Train, prep: TrainingPreparation, tokenizer: bytes, context: WorkerContext,
                 progress: Callable[[str], None]):
        """Materialize bounded package bytes without executing source or allocating model tensors yet."""
        if prep.sft is None or command.recovery is None:
            raise ValueError("SFT requires an inherited request and a complete saved model package")
        self.prep, self.tokenizer, self.context, self.progress = prep, tokenizer, context, progress
        self.saved = package_transfer.materialize(context.artifacts.verify(command.recovery), context.directory / "recovery-package")
        metadata = TrainingPublication.model_validate_json(read_bytes(self.saved.directory / "meta.json", METADATA_BYTES))
        if str(metadata.generation.model_id) != (prep.sft.request.resume_from or prep.sft.request.base_model):
            raise ValueError("SFT saved package differs from the requested source model")
        self.metadata = metadata.model_dump(mode="json")
        self.provenance: dict = {}
        self.reference = self.saved
        if prep.sft.reference is not None:
            self.reference = package_transfer.materialize(context.artifacts.verify(prep.sft.reference),
                                                         context.directory / "reference-package")
            original = TrainingPublication.model_validate_json(read_bytes(self.reference.directory / "meta.json", METADATA_BYTES))
            if str(original.generation.model_id) != prep.sft.request.base_model:
                raise ValueError("SFT reference package differs from the original base model")

    def restore(self, model: RecipeModule, definition: ResolvedDefinition) -> checkpoint.TrainState | None:
        """Load inherited weights; only exact SFT recovery carries its own optimizer/RNG/cursor forward."""
        self.progress("load")
        used = list(self.prep.sft.dataset.special_ids_seen)
        require_definition(self.saved, definition, self.tokenizer, used)
        recovering = self.prep.sft.request.resume_from is not None
        if recovering and self.saved.manifest.training_state_path is None:
            raise ValueError("SFT recovery requires a complete saved optimizer/cursor generation")
        path = self.saved.manifest.training_state_path if recovering else "model.pt"
        state = checkpoint.load_state(self.saved.directory / path, self.context.device)
        validate_saved_tensors(model, state.state_dict)
        if recovering:
            groups = optimizer_groups(model, weight_decay=self.prep.config.weight_decay)
            if (state.extra.get("training_kind") != "sft" or state.optimizer is None
                    or checkpoint.group_layout(model, groups) != state.optimizer_parameter_groups):
                raise ValueError("SFT recovery optimizer kind or ownership/order differs from the saved generation")
        model.load_state_dict(state.state_dict, strict=True)
        return replace(state, state_dict={}) if recovering else None

    def initialize(self, model: RecipeModule) -> None:
        """Apply declared warm-start once for a new SFT run and preserve explicit provenance across exact recovery."""
        cfg, dataset = self.prep.sft.request, self.prep.sft.dataset
        seen = marker_history(self.metadata, model.facts.vocabulary_size)
        used = list(dataset.special_ids_seen)
        measured = dataset.model_dump(mode="json")
        measured["special_ids_seen"] = sorted(set(used) | set(seen or []))
        measured["marker_history_complete"] = seen is not None
        if cfg.resume_from is not None:
            source = self.metadata.get("result", {})
            rows = ROW_RESULT.validate_json(encode_json(source.get("warm_start")), strict=True)
            if type(source.get("warm_started")) is not int or source["warm_started"] != len(rows.initialized_ids):
                raise ValueError("SFT recovery token initialization provenance is inconsistent")
            self.provenance = {"base_model": cfg.base_model, "dataset": measured,
                "warm_started": len(rows.initialized_ids), "warm_start": asdict(rows)}
            return
        rows = initialize_token_rows(model, untrained_marker_ids(used, seen))
        self.provenance = {"base_model": cfg.base_model, "dataset": measured,
                           "warm_started": len(rows.initialized_ids), "warm_start": asdict(rows)}

    def restore_reference(self, model: RecipeModule, definition: ResolvedDefinition) -> None:
        """Load the original frozen base, independently of the tuned/recovered weights, under the same structural contract."""
        self.progress("load")
        used = list(self.prep.sft.dataset.special_ids_seen)
        require_definition(self.reference, definition, self.tokenizer, used)
        state = checkpoint.load_state(self.reference.directory / "model.pt", self.context.device)
        validate_saved_tensors(model, state.state_dict)
        model.load_state_dict(state.state_dict, strict=True)

    def probes(self, reference: RecipeModule | None) -> Probes:
        """Verify finite held-out arrays and attach the admitted frozen model without changing training state."""
        sft = self.prep.sft
        binding = self.prep.validation.request.tokenizer
        window = self.prep.validation.request.context_window
        return Probes(pretrain=load_probe(sft.pretrain_probe, columns=window + 1,
            vocabulary_size=binding.vocabulary_size, context=self.context),
            chat=load_probe(sft.chat_probe, columns=window, vocabulary_size=binding.vocabulary_size, context=self.context),
            reference=reference)
