"""Worker-owned exact policy/reference inheritance and completed-iteration RL package publication."""

from dataclasses import replace
from collections.abc import Callable

from llm.architecture import package, package_transfer
from llm.architecture.build import RecipeModule
from llm.architecture.package_execution import validate_saved_tensors
from llm.architecture.package_integrity import METADATA_BYTES, read_bytes
from llm.architecture.spec import ResolvedDefinition
from llm.architecture.tokenizer_binding import TOKENIZER_BYTES
from llm.config import RlConfig
from llm.model import checkpoint
from llm.runtime.protocol import Train
from llm.runtime.worker import WorkerContext
from llm.train.modular_checkpoints import PackageWriter
from llm.train.modular_preflight import ReadyModel
from llm.train.modular_records import TrainingPublication
from llm.train.model_policy import optimizer_groups
from llm.train.progress import Checkpoint
from llm.train.rl_records import RlPreparation


class RlPackages:
    """Own inherited private package materializations and exact restoration through one admitted RL request."""

    def __init__(self, command: Train, prep: RlPreparation, tokenizer: bytes, context: WorkerContext,
                 progress: Callable[[str], None]):
        """Verify package identities without constructing source models or loading tensors yet."""
        if command.recovery is None:
            raise ValueError("RL requires the saved policy package")
        self.prep, self.tokenizer, self.context, self.progress = prep, tokenizer, context, progress
        self.saved = package_transfer.materialize(context.artifacts.verify(command.recovery), context.directory / "policy-package")
        self.metadata = TrainingPublication.model_validate_json(read_bytes(self.saved.directory / "meta.json", METADATA_BYTES))
        cfg = prep.request
        if str(self.metadata.generation.model_id) != (cfg.resume_from or cfg.base_model):
            raise ValueError("RL policy package differs from the requested saved model")
        self.reference = self.saved
        if prep.reference is not None:
            self.reference = package_transfer.materialize(context.artifacts.verify(prep.reference), context.directory / "reference-package")
            original = TrainingPublication.model_validate_json(read_bytes(self.reference.directory / "meta.json", METADATA_BYTES))
            if str(original.generation.model_id) != cfg.base_model:
                raise ValueError("RL reference package differs from the original base model")

    def require_definition(self, checked: package.VerifiedPackage, definition: ResolvedDefinition) -> None:
        """Require byte-identical saved structure/tokenizer and fp32 master arithmetic before loading policy or reference tensors."""
        if (checked.manifest.definition != definition or checked.manifest.parameter_precision not in (None, "fp32")
                or read_bytes(checked.directory / "tokenizer.json", TOKENIZER_BYTES) != self.tokenizer):
            raise ValueError("RL package differs from its inherited definition, tokenizer or master precision")

    def restore(self, model: RecipeModule, definition: ResolvedDefinition) -> checkpoint.TrainState | None:
        """Load policy weights and require exact optimizer ownership, original request and completed iteration for recovery."""
        self.progress("load")
        self.require_definition(self.saved, definition)
        cfg = self.prep.request
        recovering = cfg.resume_from is not None
        if recovering and self.saved.manifest.training_state_path is None:
            raise ValueError("RL recovery requires a complete saved optimizer/iteration generation")
        path = self.saved.manifest.training_state_path if recovering else "model.pt"
        state = checkpoint.load_state(self.saved.directory / path, self.context.device)
        validate_saved_tensors(model, state.state_dict)
        if recovering:
            original = RlConfig.model_validate(state.extra.get("config", {}))
            mutable = {"model_name", "resume_from", "checkpoint_target"}
            groups = optimizer_groups(model, weight_decay=cfg.weight_decay)
            if (state.extra.get("training_kind") != "rl" or state.optimizer is None
                    or checkpoint.group_layout(model, groups) != state.optimizer_parameter_groups
                    or not 0 < state.step <= cfg.iterations or state.total_steps != cfg.iterations
                    or state.step != self.metadata.generation.step or state.total_steps != self.metadata.generation.total_steps
                    or original.model_dump(exclude=mutable) != cfg.model_dump(exclude=mutable)
                    or "task_rng" not in state.extra or "sampling_rng" not in state.extra
                    or state.extra.get("curriculum") != self.metadata.result.get("curriculum")
                    or not {"torch", "numpy_keys", "numpy_position", "numpy_has_gauss", "numpy_cached_gaussian", "python"} <= state.rng.keys()):
                raise ValueError("RL recovery differs from its optimizer ownership, original request or completed iteration")
        model.load_state_dict(state.state_dict, strict=True)
        return replace(state, state_dict={}) if recovering else None

    def restore_reference(self, model: RecipeModule, definition: ResolvedDefinition) -> None:
        """Load the original frozen base independently of the tuned policy, including after recovery."""
        self.progress("load")
        self.require_definition(self.reference, definition)
        state = checkpoint.load_state(self.reference.directory / "model.pt", self.context.device)
        validate_saved_tensors(model, state.state_dict)
        model.load_state_dict(state.state_dict, strict=True)


class RlCheckpointWriter(PackageWriter):
    """Publish only completed RL iterations with independent task/sampling RNG and curriculum state."""

    def __init__(self, ready: ReadyModel, prep: RlPreparation, tokenizer: bytes, context: WorkerContext, *, provenance: dict):
        """Bind original request/reference provenance to the common bounded immutable package owner."""
        self.completed_state: dict = {}
        super().__init__(ready, tokenizer, context, request=prep.request, validation=prep.validation,
            storage=prep.storage, kind="rl", provenance=provenance)

    def recovery_extra(self, cursor: dict | None) -> dict:
        """Supply completed task/sampling/curriculum state without live gym authority or replayable trajectories."""
        if cursor is not None or not {"task_rng", "sampling_rng"} <= self.completed_state.keys():
            raise ValueError("RL checkpoint requires one completed iteration and both independent RNG states")
        return self.completed_state

    def publication_facts(self) -> dict:
        """Identify the immutable original frozen reference without fabricating corpus/cursor identities."""
        return {"base_model": self.request.base_model}

    def finish(self, model: RecipeModule, result: dict) -> None:
        """Publish the reserved complete RL output independently of intermediate checkpoint cadence."""
        frame = Checkpoint(model_id=self.request.model_name, index=0, step=result["iterations"],
            total_steps=self.request.iterations, epoch=1, stage=0, stage_name=None, path="model.pt",
            loss=result.get("final_loss"), architecture={"ctx_window": self.ready.definition.context_window}, final=True)
        self.publish(model, frame, result=result)
