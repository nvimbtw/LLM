"""Production child pretraining backend; service authorization and durable adoption stay in the parent."""

from dataclasses import replace
from collections.abc import Callable
import gc
import random
import resource

import numpy as np
from pydantic import TypeAdapter
import torch

from llm.architecture import package_transfer
from llm.architecture.build import RecipeModule
from llm.architecture.package_execution import validate_saved_tensors
from llm.architecture.package_integrity import encode_json
from llm.architecture.spec import ResolvedDefinition, canonical_digest
from llm.config import TrainConfig
from llm.model import checkpoint
from llm.runtime.protocol import Completed, Progress as RuntimeProgress, RuntimePins, Train
from llm.runtime.worker import WorkerCancelled, WorkerContext
from llm.train import loop, memory
from llm.train.modular_checkpoints import CheckpointWriter
from llm.train.modular_inputs import TrainingInputs, load_inputs
from llm.train.modular_preflight import ReadyModel, preflight
from llm.train.modular_records import TrainingPreparation
from llm.train.model_policy import optimizer_groups
from llm.train.progress import Progress
from llm.train.sft_worker import SftOperations

PROGRESS = TypeAdapter(Progress)


class Cancellation:
    """Adapt the worker's actual IPC cancellation boundary to the existing shared loop."""

    def __init__(self, context: WorkerContext):
        """Retain only the owning child context, never a parent event or device gate."""
        self.context = context

    def is_set(self) -> bool:
        """Raise the runtime cancellation signal or allow another consistent batch boundary."""
        self.context.check_cancelled()
        return False


class TrainingBackend:
    """Own source/model/optimizer work for one complete pretraining command and release it before completion."""

    def __init__(self, context: WorkerContext):
        """Bind the admitted worker context; factory import alone executes no administrator source."""
        self.context = context
        self.ready: ReadyModel | None = None

    def phase(self, value: str) -> None:
        """Report actual construction/capture/numerical phases under their finite guardian deadlines."""
        phase = "validation" if value == "numerical" else value
        self.context.progress(RuntimeProgress(type="progress", phase=phase, completed=0, unit="phases"))

    def batch(self, completed: int) -> None:
        """Refresh the training watchdog only after an actual consumed batch, including skipped masked batches."""
        self.context.progress(RuntimeProgress(type="progress", phase="backward", completed=completed, unit="batches"))

    def report(self, frame) -> None:
        """Publish bounded measured training telemetry with acknowledgement; checkpoints use their own package adoption."""
        if isinstance(frame, Progress):
            self.context.progress(RuntimeProgress(type="progress", phase="backward", completed=frame.step,
                total=frame.total_steps, unit="steps", loss=float(frame.loss)))
            artifact = self.context.artifacts.write_bytes(PROGRESS.dump_json(frame), "training_progress_v1")
            self.context.publish(artifact, completed_step=frame.step)

    def restore(self, command: Train, prep: TrainingPreparation) -> Callable[[RecipeModule, ResolvedDefinition], checkpoint.TrainState] | None:
        """Materialize one saved package and return a model/state restorer that runs after meta admission."""
        if command.recovery is None:
            if prep.config.resume_from is not None or prep.config.recover_exact:
                raise ValueError("requested training resume has no private recovery package")
            return None
        if prep.config.resume_from is None:
            raise ValueError("recovery package requires an explicit saved model identity")
        directory = self.context.directory / "recovery-package"
        checked = package_transfer.materialize(self.context.artifacts.verify(command.recovery), directory)

        def restore_model(model: RecipeModule, definition: ResolvedDefinition) -> checkpoint.TrainState:
            """Require exact saved definition/tensor/alias/group identities before loading moments or permitting updates."""
            if checked.manifest.definition != definition or checked.manifest.parameter_precision not in (None, "fp32"):
                raise ValueError("recovery package differs from the pinned fp32-master definition")
            state = checkpoint.load_state(directory / (checked.manifest.training_state_path or "model.pt"), self.context.device)
            validate_saved_tensors(model, state.state_dict)
            groups = optimizer_groups(model, weight_decay=prep.config.weight_decay)
            if state.optimizer is not None and checkpoint.group_layout(model, groups) != state.optimizer_parameter_groups:
                raise ValueError("recovery optimizer ownership/order differs from the saved generation")
            model.load_state_dict(state.state_dict, strict=True)
            return replace(state, state_dict={})
        return restore_model

    def continuation(self, prep: TrainingPreparation, inputs: TrainingInputs, cfg: TrainConfig) -> checkpoint.TrainState | None:
        """Distinguish exact data/config/schedule recovery from an ordinary weights/moments continuation."""
        if self.ready is None:
            raise RuntimeError("training continuation requires completed final preflight")
        state = self.ready.state
        if state is None:
            return None
        _, total = loop.step_plan(inputs.corpus, inputs.plan, cfg, inputs.mask)
        matching = (state.step > 0 and state.total_steps == total
            and state.extra.get("data_fingerprint") == prep.data.fingerprint)
        if prep.config.recover_exact:
            mutable = {"model_name", "resume_from", "recover_exact", "checkpoint_target", "rebuild_dataset", "architecture"}
            original = type(prep.request_config).model_validate(state.extra.get("config", {}))
            if (not matching or state.extra.get("cursor") is None
                    or state.extra.get("recomputation_policy") != self.ready.recomputation.as_dict()["selected"]
                    or original.model_dump(exclude=mutable) != prep.request_config.model_dump(exclude=mutable)):
                raise ValueError("exact recovery requires the original data, configuration, cursor and schedule")
        return state if matching else replace(state, step=0, rng={}, extra={**state.extra, "cursor": None})

    def handle(self, command, context: WorkerContext) -> Completed:
        """Verify grants, preflight weights and train/save in one admitted command without service clients.

        NumPy's global RNG takes the low 32 seed bits; Torch/data ordering retain
        the full declared seed under their own supported seed contracts.
        """
        if not isinstance(command, Train) or command.training_kind not in {"pretrain", "sft"} or command.pins is not None:
            raise ValueError("training backend requires one empty-session pretrain or inherited SFT command")
        if command.job_id != context.request_id:
            raise ValueError("training command belongs to another runtime request")
        prep = TrainingPreparation.model_validate_json(context.artifacts.read_bytes(command.config))
        if prep.training_kind != command.training_kind:
            raise ValueError("training command kind differs from its typed preparation")
        if prep.validation.request.profile.device != context.device:
            raise ValueError("training profile differs from the admitted worker device")
        self.phase("load")
        inputs = load_inputs(command, prep, context)
        cfg = prep.config.model_copy(update={"ctx_window": prep.validation.request.context_window})
        if context.device == "cuda" and cfg.compile_model and cfg.recompute_blocks not in (None, False, 0):
            raise ValueError("explicit recipe replay requires compile_model=false until checkpoint compilation is validated")
        _steps, total = loop.step_plan(inputs.corpus, inputs.plan, cfg, inputs.mask)
        if any(stage.epochs < 1 for stage in inputs.plan.ordered_stages(cfg.epochs)) or not 1 <= total < 2**53:
            raise ValueError("training data and positive epoch schedule must yield at least one full batch")
        torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32 if context.device == "cuda" else False
        torch.backends.cudnn.allow_tf32 = cfg.allow_tf32 if context.device == "cuda" else False
        random.seed(cfg.seed)
        np.random.seed(cfg.seed % 2**32)
        baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        resources = context.bootstrap.policy.resources
        allocator = (min(resources.device_allocator_bytes, memory.budget_bytes("cuda")) if context.device == "cuda"
                     else resources.tree_resident_bytes - baseline)
        sft = SftOperations(command, prep, inputs.tokenizer, context, self.phase) if prep.sft is not None else None
        probes = sft.probes(None) if sft is not None else None
        restore = sft.restore if sft is not None else self.restore(command, prep)
        reference = sft.restore_reference if sft is not None and prep.sft.chat_probe is not None else None
        self.ready = preflight(prep.validation, allocator_bytes=allocator, progress=self.phase,
                               restore=restore, recompute=cfg.recompute_blocks,
                               recover_exact=cfg.recover_exact, compiled=context.device == "cuda" and cfg.compile_model,
                               initialize=sft.initialize if sft is not None else None, restore_reference=reference)
        if probes is not None:
            probes.reference = self.ready.reference
        receipt = self.ready.validation.report.receipt
        context.pins = RuntimePins(definition_id=canonical_digest(self.ready.definition), bindings_digest=receipt.identity.parameter_digest,
            profile_digest=canonical_digest(receipt.identity.profile), receipt_id=receipt.receipt_id)
        state = self.continuation(prep, inputs, cfg)
        provenance = sft.provenance if sft is not None else ({"dataset": {
            "special_ids_seen": list(prep.data.special_ids_seen), "marker_history_complete": True}}
            if prep.data.special_ids_seen is not None else {})
        writer = CheckpointWriter(self.ready, prep, inputs.tokenizer, context, provenance=provenance)
        self.batch(0)
        result = loop.run(self.ready.model, inputs.corpus, inputs.plan, cfg, context.device,
            on_progress=self.report, cancel=Cancellation(context), token_counts=prep.data.token_counts,
            on_checkpoint=writer, resume=state, model_plan=self.ready.memory, on_batch=self.batch,
            mask=inputs.mask, probes=probes)
        result.update({**provenance, "model_id": cfg.model_name, "definition_id": canonical_digest(self.ready.definition),
            "vocab": self.ready.definition.tokenizer.vocabulary_size, "ctx_window": cfg.ctx_window,
            "parameters": self.ready.validation.report.unique_parameter_count, "corpus_tokens": prep.data.tokens,
            "recomputation": self.ready.recomputation.as_dict(),
            "documents": len(inputs.plan.documents), "data_fingerprint": prep.data.fingerprint,
            "tokenizer_id": str(self.ready.definition.tokenizer.tokenizer_id), "corpus_key": prep.data.corpus_key})
        if result["cancelled"]:
            raise WorkerCancelled("training stopped before complete data consumption")
        writer.finish(self.ready.model, result)
        artifact = context.artifacts.write_bytes(encode_json(result), "training_result_v1")
        return Completed(type="completed", outcome="ok", result=artifact)

    def finish_request(self) -> None:
        """Drop compiler/model/import/optimizer state before the runtime may emit successful completion."""
        torch._dynamo.reset()
        if self.ready is not None:
            self.ready.close()
            self.ready = None
        self.context.pins = None
        gc.collect()
        if self.context.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self) -> None:
        """Release the complete training lifetime on success, cancellation or failure."""
        self.finish_request()
        self.context = None
