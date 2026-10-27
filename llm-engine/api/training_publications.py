"""Parent-only verified package adoption, durable training evidence and immutable model registration."""

from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
from uuid import UUID
from sqlalchemy.orm import Session

import store
from api import sft_authority
from api.training_jobs import frozen_recipe, inherited_pin
from llm import storage
from llm.architecture import package, package_transfer
from llm.architecture.package_integrity import METADATA_BYTES, PackageBudget, publish_directory, read_bytes, sync_directory
from llm.architecture.spec import canonical_digest
from llm.config import RlConfig, SftConfig, TrainConfig
from llm.runtime.artifacts import ArtifactStore
from llm.runtime.protocol import ArtifactReady
from llm.train.model_preflight import rl_profile, sft_loop_config, training_profile
from llm.train.modular_records import TrainingGeneration, TrainingPublication
from llm.train.progress import Checkpoint
from store import architecture_queries as catalog, queries as q
from store.schema import Run, RunStatus


class ObsoleteGeneration(ValueError):
    """A verified unregistered recovery generation is superseded by durable same-run output."""


def metadata(checked: package.VerifiedPackage) -> TrainingPublication:
    """Read bounded typed generation metadata and bind interpreted facts to its verified package."""
    value = TrainingPublication.model_validate_json(read_bytes(checked.directory / "meta.json", METADATA_BYTES))
    generation, definition = value.generation, checked.manifest.definition
    if (generation.architecture != {"ctx_window": definition.context_window}
            or (checked.manifest.training_state_path is None) != generation.final
            or checked.manifest.parameter_precision != "fp32"
            or (not generation.final and not checked.manifest.optimizer_parameter_order)
            or value.result.get("recovery") != (not generation.final)
            or TrainingGeneration.model_validate(value.result.get("checkpoint")) != generation
            or generation.skipped_batches != 0 and value.training_kind != "sft"
            or value.result.get("definition_id") != checked.manifest.definition_id
            or value.result.get("vocab") != definition.tokenizer.vocabulary_size
            or value.result.get("ctx_window") != definition.context_window
            or type(value.result.get("parameters")) is not int or value.result["parameters"] < 0):
        raise ValueError("training publication metadata differs from its verified package")
    return value


def validate_job(db: Session, checked: package.VerifiedPackage, value: TrainingPublication) -> Run:
    """Check exact queued config/pins/profile and reserved output before any package can enter model storage."""
    run = q.lock_run(db, value.job_id)
    pin = inherited_pin(db, value.job_id)
    if run is None or run.training is None or pin is None or run.training.training_type != value.training_kind:
        raise ValueError("training publication has no matching pinned training job")
    if run.status in (RunStatus.queued, RunStatus.waiting):
        raise ValueError("training publication belongs to a job that was never admitted")
    cfg = {"pretrain": TrainConfig, "sft": SftConfig, "rl": RlConfig}[value.training_kind].model_validate(run.config)
    if cfg.model_dump() != value.config.model_dump() or run.training.target_model_id != UUID(cfg.model_name):
        raise ValueError("training publication differs from its queued configuration/output")
    generation, definition = value.generation, checked.manifest.definition
    if (generation.model_id == run.training.target_model_id) != generation.final:
        raise ValueError("training generation uses the wrong reserved final/recovery identity")
    _recipe, context, _names = frozen_recipe(pin)
    if definition.context_window != context or definition.architecture != pin.evidence.architecture or definition.parameters != pin.parameters:
        raise ValueError("training publication differs from frozen recipe/parameter authority")
    if value.training_kind == "sft":
        sft_authority.validate(db, checked, value, cfg)
    if isinstance(cfg, RlConfig):
        validate_rl_source(db, checked, value, cfg)
    effective = sft_loop_config(cfg, context) if isinstance(cfg, SftConfig) else cfg
    for receipt in checked.receipts:
        profile = receipt.identity.profile
        expected = rl_profile(cfg, context, profile.device) if isinstance(cfg, RlConfig) else training_profile(effective, context, profile.device)
        allowed_precision = {"fp32"} if profile.device == "cpu" or isinstance(cfg, RlConfig) else ({"bf16", "fp16"} if cfg.precision == "bf16" else {cfg.precision})
        if (profile.precision not in allowed_precision
                or profile.model_copy(update={"export_guards": (), "precision": expected.precision}) != expected
                or receipt.identity.input_profile_digest != canonical_digest({"tensor_inputs": {},
                    "domains": [domain.model_dump(mode="json") for domain in expected.domains]})):
            raise ValueError("training publication differs from its effective input/runtime profile")
    if len(checked.receipts) != 1:
        raise ValueError("one training job owns exactly one final-preflight receipt")
    return run


def validate_rl_source(db: Session, checked: package.VerifiedPackage, value: TrainingPublication, cfg: RlConfig) -> None:
    """Bind RL publication to registered policy/original-reference definitions and the exact queued iteration schedule.

    A waiting chain may retain its original pending tokenizer pin; actual saved
    model identity supplies final inherited authority without rewriting that pin
    or requiring surviving predecessor job history.
    """
    for identity in {cfg.resume_from or cfg.base_model, cfg.base_model}:
        source = q.get_model(db, UUID(identity))
        if source is None or source.definition_id != checked.manifest.definition_id:
            raise ValueError("RL publication differs from its registered policy/original-reference definition")
    if value.generation.total_steps != cfg.iterations or value.result.get("base_model") != cfg.base_model:
        raise ValueError("RL publication differs from its original reference or queued iteration schedule")


def register(checked: package.VerifiedPackage) -> Checkpoint:
    """Register complete parent-owned bytes and retain matching evidence before retiring older recovery storage.

    Database failure preserves the immutable orphan for startup adoption and
    propagates to stop further publication. Repeated registration preserves
    existing model display/publication/keep metadata. No custom source or tensor
    deserialization occurs in this parent adapter.
    """
    value = metadata(checked)
    frame = value.generation.checkpoint()
    if checked.directory != storage.model_dir(frame.model_id):
        raise ValueError("training registration requires its authoritative immutable model directory")
    retired = []
    with store.session() as db:
        catalog.bound_training_publication(db)
        existing = q.get_model(db, value.generation.model_id)
        if existing is not None:
            if existing.definition_id != checked.manifest.definition_id or existing.run_id != value.job_id:
                raise ValueError("model UUID already identifies another training publication")
            return replace(frame, path=str(checked.directory / frame.path))
        run = validate_job(db, checked, value)
        if not frame.final:
            siblings = q.checkpoints_of(db, run.training.target_model_id)
            if (q.get_model(db, run.training.target_model_id) is not None
                    or any(row.checkpoint_step >= frame.step or row.checkpoint_index >= frame.index for row in siblings)):
                raise ObsoleteGeneration("recovery generation is older than registered same-run output")
        receipt = checked.receipts[0]
        catalog.record_training_evidence(db, run.id, receipt, checked.manifest.definition,
            prepared_tokenizer=checked.manifest.definition.tokenizer if value.training_kind == "sft" else None)
        definition = catalog.record_definition(db, checked.manifest.definition, checked.manifest.receipt_ids)
        if run.training.definition_id not in (None, definition.id):
            raise ValueError("training run already owns another resolved definition")
        run.training.definition_id = definition.id
        config = run.config
        display = config.get("display_name") or config["model_name"]
        name = display if frame.final else f"{str(display)[:q.NAME_LIMIT - 8].rstrip()} #{frame.index}"
        q.upsert_model(db, model_id=value.generation.model_id, name=name, published=False, run_id=run.id,
            parent_id=None if frame.final else run.training.target_model_id,
            checkpoint_step=None if frame.final else frame.step, checkpoint_index=None if frame.final else frame.index,
            definition_id=definition.id, context=checked.manifest.definition.context_window, family=value.training_kind,
            parameters={"config": config, "result": {**value.result, "output_model_id": config["model_name"]}})
        if not frame.final:
            retired = q.retire_recovery(db, run.training.target_model_id, value.generation.model_id)
    for model_id in retired:
        directory = storage.model_dir(str(model_id))
        if directory.exists():
            shutil.rmtree(directory)
    return replace(frame, path=str(checked.directory / frame.path))


def adopt(notice: ArtifactReady, artifacts: ArtifactStore, *, job_id: UUID,
          budget: PackageBudget = PackageBudget()) -> Checkpoint:
    """Materialize one verified grant, publish immutable model bytes and commit registration before returning for ACK."""
    if notice.artifact.format != "model_package_v1":
        raise ValueError("training model adoption requires a model package grant")
    root = storage.package_staging_dir()
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < notice.artifact.size_bytes + (64 << 20):
        raise ValueError("training publication staging has insufficient free space")
    staging = Path(tempfile.mkdtemp(prefix="training-", dir=root))
    try:
        checked = package_transfer.materialize(artifacts.verify(notice.artifact), staging / "package", budget=budget)
        value = metadata(checked)
        if value.job_id != job_id or value.generation.step != notice.completed_step or value.generation.final != notice.final:
            raise ValueError("training publication notice differs from its complete generation")
        with store.session() as db:
            catalog.bound_training_publication(db)
            validate_job(db, checked, value)
        destination = storage.model_dir(str(value.generation.model_id))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            previous = package.verify(destination, budget=budget)
            if previous.integrity != checked.integrity or previous.manifest != checked.manifest:
                raise ValueError("model UUID already contains different immutable package bytes")
        else:
            publish_directory(checked.directory, destination)
            sync_directory(destination.parent)
            sync_directory(staging)
        return register(package.verify(destination, budget=budget))
    finally:
        shutil.rmtree(staging)


def reconcile(directory: Path) -> None:
    """Adopt complete interrupted publications after exclusive worker cleanup; remove superseded unregistered recovery copies."""
    checked = package.verify(directory)
    value = metadata(checked)
    if directory != storage.model_dir(str(value.generation.model_id)):
        raise ValueError("training package directory differs from its generation UUID")
    try:
        register(checked)
    except ObsoleteGeneration:
        shutil.rmtree(directory)
        sync_directory(directory.parent)
