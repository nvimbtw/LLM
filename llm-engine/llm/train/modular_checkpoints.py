"""Worker-local immutable checkpoint packages and acknowledged parent publication."""

from dataclasses import asdict
import shutil
from typing import Literal
from uuid import uuid4

from llm.architecture import package, package_transfer
from llm.architecture.package_integrity import PackageBudget, decode_json, encode_json
from llm.architecture.spec import DefinitionPackage, canonical_digest
from llm.architecture.validation_records import ValidationSnapshot
from llm.config import RlConfig, SftConfig, TrainConfig
from llm.model import checkpoint
from llm.runtime.worker import WorkerContext
from llm.train.modular_preflight import ReadyModel
from llm.train.modular_records import TrainingPreparation
from llm.train.modular_records import TrainingGeneration, TrainingPublication, TrainingStorage
from llm.train.progress import Checkpoint
from llm.train.rl_records import MAX_RL_HISTORY_BYTES


class PackageWriter:
    """Own numerical training package serialization and acknowledged adoption across supervised and reinforcement learning."""

    def __init__(self, ready: ReadyModel, tokenizer: bytes, context: WorkerContext, *,
                 request: TrainConfig | SftConfig | RlConfig, validation: ValidationSnapshot,
                 storage: TrainingStorage, kind: Literal["pretrain", "sft", "rl"], provenance: dict | None = None):
        """Retain exact source/evidence/tokenizer identities beside the live model/import lifetime."""
        self.ready, self.tokenizer, self.context = ready, tokenizer, context
        self.request, self.validation, self.kind = request, validation, kind
        self.provenance = dict(provenance or {})
        self.index = 0
        overhead = package_transfer.archive_limit(PackageBudget(total_bytes=1, file_bytes=1)) - 1
        total = min(storage.package_bytes, context.bootstrap.policy.resources.artifact_bytes - overhead)
        self.budget = PackageBudget(total_bytes=total, file_bytes=min(total, storage.file_bytes))
        self.generation_bytes = self.estimate_bytes()
        self.require_capacity()

    def estimate_bytes(self) -> int:
        """Preflight owned weights, AdamW moments and bounded serialization/source/graph metadata before optimization.

        Serialized files additionally enforce exact write quotas. The metadata
        allowance is conservative policy for the pinned PyTorch format, not an
        assertion that arbitrary hidden custom objects can be checkpointed.
        """
        tensors = self.ready.model.state_dict()
        names = tuple(name for name, parameter in self.ready.model.named_parameters() if parameter.requires_grad)
        if len(names) > 4096 or any(len(name) > 512 for name in names):
            raise ValueError("training optimizer ownership exceeds saved-package limits")
        metadata = (1 << 20) + sum(len(name.encode()) + 4096 for name in tensors)
        weights = sum(value.numel() * value.element_size() for value in tensors.values()) + metadata
        moments = 2 * sum(value.numel() * value.element_size() for value in self.ready.model.parameters() if value.requires_grad)
        state = moments + 2 * metadata + len(encode_json(
            self.request.model_dump(mode="json", exclude_unset=True))) + len(encode_json(self.provenance))
        if self.kind == "rl":
            state += MAX_RL_HISTORY_BYTES
        if self.request.checkpoint_target == 0:
            state = 0
        static = len(self.tokenizer) + len(self.validation.recipe.encode()) + sum(map(len, self.ready.sources.values()))
        static += sum(len(value.encode()) for value in self.ready.validation.graph_blobs.values())
        static += len(encode_json(self.ready.validation.report.receipt)) + len(encode_json(self.ready.definition)) + (2 << 20)
        static += len(encode_json(self.provenance))
        if self.kind == "rl":
            static += MAX_RL_HISTORY_BYTES
        total = weights + state + static
        if max(weights, state) > self.budget.file_bytes or total > self.budget.total_bytes:
            raise ValueError("training generation does not fit admitted checkpoint byte budget")
        return total

    def require_capacity(self) -> None:
        """Require space for generation files, sealed package, archive and grant before writing another checkpoint."""
        if shutil.disk_usage(self.context.directory).free < 4 * self.generation_bytes + (64 << 20):
            raise ValueError("training checkpoint staging has insufficient free space")

    def __call__(self, model, optimizer, *, step: int, total_steps: int, epoch: int, stage: int,
                 stage_name: str | None, loss: float | None, cursor: dict | None = None) -> Checkpoint:
        """Write matching tensor/optimizer/RNG/cursor files and publish one complete automatic recovery generation."""
        self.index += 1
        frame = Checkpoint(model_id=str(uuid4()), index=self.index, step=step, total_steps=total_steps,
            epoch=epoch, stage=stage, stage_name=stage_name, path="train_state.pt", loss=loss,
            architecture={"ctx_window": self.ready.definition.context_window})
        self.publish(model, frame, optimizer=optimizer, cursor=cursor)
        return frame

    def publish(self, model, frame: Checkpoint, *, optimizer=None, cursor: dict | None = None,
                result: dict | None = None) -> None:
        """Seal a self-contained package in session storage, grant its USTAR container, then retire all local copies."""
        root = self.context.directory / f"generation-{frame.model_id}"
        self.require_capacity()
        root.mkdir()
        archive = root / "package.tar"
        try:
            definition_id = canonical_digest(self.ready.definition)
            weights = checkpoint.save_modular_weights(model, root / "model.pt", definition_id, max_bytes=self.budget.file_bytes)
            files = self.static_files(frame, result)
            files["model.pt"] = weights
            order = ()
            if optimizer is not None:
                files["train_state.pt"] = checkpoint.save_modular_state(root / "train_state.pt", model=model,
                    optimizer=optimizer, definition_id=definition_id, step=frame.step, epoch=frame.epoch,
                    stage=frame.stage, total_steps=frame.total_steps, rng=checkpoint.rng_state(),
                    extra=self.state_extra(cursor), max_bytes=self.budget.file_bytes)
                order = tuple(name for group in checkpoint.optimizer_layout(model, optimizer) for name in group)
            receipt = self.ready.validation.report.receipt
            manifest = DefinitionPackage(format="slate.modular-model", format_version=1,
                definition_id=definition_id, definition=self.ready.definition, receipt_ids=(receipt.receipt_id,),
                graph_hashes=tuple(self.ready.validation.graph_blobs), parameter_precision="fp32",
                training_state_path="train_state.pt" if optimizer is not None else None, optimizer_parameter_order=order)
            destination = root / "package"
            package.publish(destination, manifest, files, staging_root=root / "staging", budget=self.budget)
            package_transfer.write_archive(destination, archive, budget=self.budget)
            grant = self.context.artifacts.stage_file(archive, "model_package_v1")
            self.context.publish(grant, completed_step=frame.step, final=frame.final)
        finally:
            shutil.rmtree(root)

    def state_extra(self, cursor: dict | None) -> dict:
        """Freeze the exact original request/replay identity and domain-specific completed-boundary state beside optimizer/global RNG."""
        return {"training_kind": self.kind,
                "config": self.request.model_dump(mode="json", exclude_unset=True),
                "recomputation_policy": self.ready.recomputation.as_dict()["selected"],
                **self.recovery_extra(cursor),
                **({"provenance": self.provenance} if self.provenance else {})}

    def recovery_extra(self, cursor: dict | None) -> dict:
        """Supply the owning training domain's completed-boundary recovery state for serialization."""
        raise NotImplementedError

    def publication_facts(self) -> dict:
        """Supply the owning domain's bounded measured input facts for parent-readable publication metadata."""
        raise NotImplementedError

    def static_files(self, frame: Checkpoint, result: dict | None) -> dict:
        """Build canonical source/evidence/metadata files without tensor serialization or parent storage access."""
        receipt = self.ready.validation.report.receipt
        skipped = result["total_steps"] - result["steps"] if frame.final and self.kind == "sft" else 0
        generation = TrainingGeneration.model_validate({**asdict(frame), "skipped_batches": skipped})
        files = {"architecture.yaml": self.validation.recipe.encode("utf-8"), "tokenizer.json": self.tokenizer,
            "training-plan.json": encode_json({"definition_id": canonical_digest(self.ready.definition),
                "receipt_id": str(receipt.receipt_id), "memory": self.ready.memory.as_dict(),
                "recomputation": self.ready.recomputation.as_dict()}),
            "meta.json": TrainingPublication(job_id=self.validation.run_id, training_kind=self.kind,
                config=self.request,
                result=decode_json(encode_json({**self.provenance, **(result or {}), "definition_id": canonical_digest(self.ready.definition),
                    "vocab": self.ready.definition.tokenizer.vocabulary_size,
                    "ctx_window": self.ready.definition.context_window,
                    "parameters": self.ready.validation.report.unique_parameter_count,
                    **self.publication_facts(),
                    "recomputation": self.ready.recomputation.as_dict(),
                    "recovery": not frame.final, "checkpoint": generation.model_dump(mode="json")})),
                generation=generation).model_dump_json().encode(),
            f"receipts/{receipt.receipt_id}.json": encode_json(receipt)}
        files.update({entry.path: self.ready.sources[entry.reference.revision_id] for entry in self.ready.definition.dependencies})
        files.update({f"graphs/{digest}.json": source.encode("utf-8") for digest, source in self.ready.validation.graph_blobs.items()})
        return files


class CheckpointWriter(PackageWriter):
    """Bind packed-data cursors and final schedule metadata to the common acknowledged package publisher."""

    def __init__(self, ready: ReadyModel, prep: TrainingPreparation, tokenizer: bytes, context: WorkerContext,
                 *, provenance: dict | None = None):
        """Retain the exact prepared corpus/SFT authority through its model and publication lifetime."""
        self.prep = prep
        super().__init__(ready, tokenizer, context, request=prep.request_config, validation=prep.validation,
            storage=prep.storage, kind=prep.training_kind, provenance=provenance)

    def finish(self, model, result: dict) -> None:
        """Always publish the reserved finished output, independently of automatic checkpoint cadence."""
        stages = self.prep.data.stages
        frame = Checkpoint(model_id=self.prep.config.model_name, index=0, step=result["steps"],
            total_steps=result["total_steps"], epoch=sum(value.epochs for value in stages) if stages else self.prep.config.epochs,
            stage=len(stages) - 1 if stages else 0, stage_name=stages[-1].name if stages else None, path="model.pt",
            loss=result["final_loss"], architecture={"ctx_window": self.ready.definition.context_window}, final=True)
        self.publish(model, frame, result=result)

    def recovery_extra(self, cursor: dict | None) -> dict:
        """Bind moments/RNG to the exact packed-data cursor and immutable corpus identity."""
        return {"cursor": cursor, **self.publication_facts()}

    def publication_facts(self) -> dict:
        """Expose measured packed-data identities without storing source documents in model metadata."""
        return {"data_fingerprint": self.prep.data.fingerprint, "corpus_key": self.prep.data.corpus_key}
