"""Existing-queue pretraining adapter: authorized parent data, one admitted child and acknowledged durable output."""

from pathlib import Path
from dataclasses import dataclass
from threading import Event
import tempfile
from uuid import UUID

from pydantic import TypeAdapter
import torch

import store
from api import training_admission, training_jobs, training_publications
from api.worker import RegisteredCheckpoint, SupervisedRun
from llm import storage
from llm.architecture import package, package_transfer
from llm.architecture.package_integrity import PackageBudget, decode_json
from llm.architecture.spec import Recipe, RevisionRef
from llm.architecture.validation_records import ValidationSnapshot
from llm.config import LEGACY_STRUCTURE_FIELDS, SftConfig, TrainConfig
from llm.corpus import vocab
from llm.orchestrate import modular_data, modular_sft_data
from llm.runtime.artifacts import ArtifactStore
from llm.runtime.protocol import ArtifactReady, Train, require_format
from llm.runtime.supervisor import RequestCancelled
from llm.train.modular_records import TrainingPreparation
from llm.train.progress import Checkpoint, Progress
from store import architecture_queries as catalog, queries as q
from store.schema import RunStatus

PROGRESS = TypeAdapter(Progress)


@dataclass(frozen=True)
class PreparedSftRequest:
    """Carry actual SFT inputs and an optional separate original KL reference under the common session owner."""

    config: SftConfig
    data: modular_sft_data.PreparedSftData
    reference: package.VerifiedPackage | None = None


@dataclass(frozen=True)
class PreparedTraining:
    """Own one parent-prepared request's immutable bindings, local inputs and admitted worker storage policy."""

    config: TrainConfig
    snapshot: ValidationSnapshot
    data: modular_data.PreparedData
    saved: package.VerifiedPackage | None
    admission: training_admission.Admission
    sft: PreparedSftRequest | None = None


def check_cancelled(cancel: Event) -> None:
    """Refuse further acquisition/publication completion once this durable request has been cancelled."""
    if cancel.is_set():
        raise RequestCancelled("pretraining request was cancelled")


def saved_package(cfg: TrainConfig, pin: catalog.PreparedTrainingPin) -> package.VerifiedPackage | None:
    """Read only registered saved package bytes and require the inherited exact definition before data acquisition."""
    if cfg.resume_from is None:
        if cfg.recover_exact:
            raise ValueError("exact recovery requires a saved model")
        return None
    if cfg.model_fields_set & LEGACY_STRUCTURE_FIELDS:
        raise ValueError("saved recipe training inherits its structural definition")
    model_id = UUID(cfg.resume_from)
    checked = package.verify(storage.model_dir(str(model_id)))
    with store.session() as db:
        model = q.get_model(db, model_id)
        if model is None or model.definition_id != checked.manifest.definition_id:
            raise ValueError("saved recipe has no matching registered model definition")
    definition = checked.manifest.definition
    expected = {item["reference"]["revision_id"]: RevisionRef.model_validate(item["reference"]) for item in pin.snapshot["dependencies"]}
    actual = {str(item.reference.revision_id): item.reference for item in definition.dependencies}
    if (definition.architecture != pin.evidence.architecture or definition.parameters != pin.parameters
            or definition.recipe != Recipe.model_validate(pin.snapshot["recipe"]) or actual != expected):
        raise ValueError("saved recipe differs from the immutable inherited training pin")
    if cfg.recover_exact and checked.manifest.training_state_path is None:
        raise ValueError("exact recovery requires a complete saved training generation")
    return checked


def record_tokenizer(cfg: TrainConfig, prepared: modular_data.PreparedData) -> None:
    """Index a newly derived actual build; resumed self-contained vocabularies need no surviving named build."""
    if cfg.resume_from is not None or cfg.tokenizer_id is not None:
        return
    identity = str(prepared.binding.tokenizer_id)
    info = vocab.info(identity)
    if info is None or info.get("fingerprint") != prepared.binding.fingerprint[:vocab.FINGERPRINT_CHARS]:
        raise ValueError("derived tokenizer has no matching complete owned build")
    with store.session() as db:
        if db.get(store.Tokenizer, prepared.binding.tokenizer_id) is None:
            q.upsert_tokenizer(db, tokenizer_id=prepared.binding.tokenizer_id, name=f"auto {identity[:8]}",
                implicit=True, vocab_size=prepared.binding.vocabulary_size, fingerprint=info["fingerprint"],
                merges=info.get("merges"), max_vocab=info.get("max_vocab"), datasets=info.get("datasets"))


class TrainingPublications:
    """Own one request's sequential adoption/telemetry callbacks and retained final output boundary."""

    def __init__(self, context: SupervisedRun, budget: PackageBudget):
        """Bind the actual parent job and finite copy policy, without retaining source/model execution objects."""
        self.context, self.budget = context, budget
        self.final: Checkpoint | None = None

    def __call__(self, notice: ArtifactReady, artifacts: ArtifactStore) -> None:
        """Commit complete model generations or forward validated measured progress before the supervisor ACK."""
        if notice.artifact.format == "training_progress_v1":
            frame = PROGRESS.validate_json(artifacts.read_bytes(notice.artifact))
            if notice.final or frame.step != notice.completed_step:
                raise ValueError("training telemetry notice differs from its measured frame")
            self.context.progress(frame)
        else:
            if self.final is not None:
                raise ValueError("training worker published after its final output")
            frame = training_publications.adopt(notice, artifacts, job_id=self.context.job_id, budget=self.budget)
            if frame.final:
                self.final = frame
            self.context.progress(RegisteredCheckpoint(frame))


def completed_result(result: dict, cfg: TrainConfig | SftConfig, prepared: modular_data.PreparedData, final: Checkpoint | None) -> dict:
    """Require the completed summary to name the durable final generation and exact prepared inputs after cleanup."""
    if (final is None or result.get("model_id") != cfg.model_name or final.model_id != cfg.model_name
            or type(result.get("steps")) is not int or result["steps"] != final.step
            or type(result.get("total_steps")) is not int or result["total_steps"] != final.total_steps
            or result.get("data_fingerprint") != prepared.fingerprint):
        raise ValueError("training completion differs from its registered final output")
    with store.session() as db:
        model = q.get_model(db, UUID(cfg.model_name))
        if model is None or model.definition_id != result.get("definition_id"):
            raise ValueError("training completion has no matching durable model definition")
    return result


class TrainingHandler:
    """Use the current runner lease for one entire parent-data/child-pretrain/package-cleanup operation."""

    def run_supervised(self, context: SupervisedRun) -> dict:
        """Finalize exact inputs, run only the trusted child factory and return after verified session cleanup."""
        cfg = TrainConfig.model_validate(context.config)
        training_admission.validate_configuration(cfg)
        with store.session() as db:
            pin = training_jobs.inherited_pin(db, context.job_id)
            run = q.lock_run(db, context.job_id)
            if (pin is None or run is None or run.training is None or run.training.training_type != "pretrain"
                    or run.status not in (RunStatus.running, RunStatus.cancelling)):
                raise ValueError("supervised pretraining requires the actual admitted pinned job")
        check_cancelled(context.cancel)
        saved = saved_package(cfg, pin)
        _recipe, window, names = training_jobs.frozen_recipe(pin)
        root = storage.package_staging_dir()
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="training-inputs-", dir=root) as temporary:
            directory = Path(temporary)
            budget = training_admission.packing_budget(training_admission.capacity(directory))
            try:
                prepared = modular_data.prepare(cfg, context_window=window, reserved_names=names, directory=directory,
                    actor=context.actor, progress=context.progress, cancel=context.cancel, saved=saved, budget=budget)
            except InterruptedError as error:
                if context.cancel.is_set():
                    raise RequestCancelled("pretraining cancelled during parent data preparation") from error
                raise
            check_cancelled(context.cancel)
            record_tokenizer(cfg, prepared)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            snapshot = training_jobs.frozen_validation(pin, cfg, prepared.binding, context.job_id, device)
            available = training_admission.capacity(directory)
            runtime_disk = training_admission.capacity(context.workers.root).disk_bytes
            available = training_admission.Capacity(available.host_bytes, min(available.disk_bytes, runtime_disk))
            admission = training_admission.admit(prepared, saved, available, device)
            request = PreparedTraining(cfg, snapshot, prepared, saved, admission)
            result = self.execute(context, request, directory)
        check_cancelled(context.cancel)
        return result

    def execute(self, context: SupervisedRun, request: PreparedTraining, directory: Path) -> dict:
        """Execute one full pretrain/SFT command with private grants under the existing lease and publication owner."""
        recovery = None
        reference = None
        cfg, prepared, admission = request.config, request.data, request.admission
        publications = TrainingPublications(context, admission.storage.package_budget())
        if request.saved is not None:
            recovery = package_transfer.write_archive(request.saved.directory, directory / "recovery.tar")
        if request.sft is not None and request.sft.reference is not None:
            reference = package_transfer.write_archive(request.sft.reference.directory, directory / "reference.tar")
        with context.workers.session(context.lease, admission.policy, "llm.train.modular_worker:TrainingBackend") as session:
            supervised = request.sft.data.grant_inputs(session.artifacts, request.sft.config,
                session.artifacts.stage_file(reference, "model_package_v1") if reference is not None else None) if request.sft else None
            prep = TrainingPreparation(config=cfg, validation=request.snapshot, storage=admission.storage,
                tokenizer=session.artifacts.stage_file(prepared.tokenizer, "tokenizer_v1"),
                data=prepared.grant_plan(session.artifacts), sft=supervised)
            payload = prep.model_dump_json().encode()
            if len(payload) > training_admission.PREPARATION_BYTES:
                raise ValueError("training preparation exceeds the bounded config grant")
            command = Train(type="train", job_id=context.job_id, training_kind=prep.training_kind,
                config=session.artifacts.write_bytes(payload, "training_config_v1"),
                corpus=session.artifacts.stage_file(prepared.corpus, "corpus_v1"),
                recovery=session.artifacts.stage_file(recovery, "model_package_v1") if recovery is not None else None)
            result = session.run(command, request_id=context.job_id, adopt=publications, cancel=context.cancel)
            if result.outcome != "ok" or result.result is None:
                raise RequestCancelled("pretraining worker did not complete")
            require_format(result.result, "training_result_v1")
            value = decode_json(session.artifacts.read_bytes(result.result))
            if not isinstance(value, dict):
                raise ValueError("training result must be a bounded JSON summary")
        check_cancelled(context.cancel)
        return completed_result(value, request.sft.config if request.sft else cfg, prepared, publications.final)
