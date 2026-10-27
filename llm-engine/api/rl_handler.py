"""Existing-queue RL adapter: immutable saved inputs, parent-authorized gym effects and one supervised numerical lifetime."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import tempfile
from uuid import UUID

import store
import torch

from api import rl_admission, rl_recovery, training_admission, training_jobs, training_publications
from api.rl_effect_log import Authority, GymEffectLog
from api.rl_effects import GymDispatcher
from api.training_handler import TrainingPublications, check_cancelled, saved_package
from api.worker import SupervisedRun
from llm import storage
from llm.architecture import package, package_transfer
from llm.architecture.package_integrity import decode_json, encode_json
from llm.architecture.validation_records import ValidationSnapshot
from llm.config import RlConfig, TrainConfig
from llm.runtime.effects import EffectBroker, EffectJournal
from llm.runtime.protocol import Train, require_format
from llm.runtime.supervisor import CleanupUncertain, RequestCancelled
from llm.train.progress import Checkpoint
from llm.train.modular_records import TrainingPublication
from llm.train.rl_records import RlPreparation, validate_configuration
from store import architecture_queries as catalog, queries as q
from store.schema import RunKind, RunStatus


@dataclass(frozen=True)
class PreparedRl:
    """Parent-owned exact source/reference packages, frozen validation and separately admitted numerical/effect storage."""

    config: RlConfig
    snapshot: ValidationSnapshot
    saved: package.VerifiedPackage
    reference: package.VerifiedPackage | None
    first_iteration: int
    training_group_start: int
    admission: training_admission.Admission
    effects: rl_admission.EffectBudget


def source_packages(cfg: RlConfig, pin: catalog.PreparedTrainingPin) -> tuple[package.VerifiedPackage, package.VerifiedPackage, int]:
    """Verify registered policy/reference bytes and original completed schedule without constructing source or loading tensors."""
    source = saved_package(TrainConfig(model_name=cfg.model_name, resume_from=cfg.resume_from or cfg.base_model,
        recover_exact=cfg.resume_from is not None), pin)
    base = saved_package(TrainConfig(model_name=cfg.model_name, resume_from=cfg.base_model), pin) if cfg.resume_from else source
    if source is None or base is None:
        raise ValueError("RL requires actual saved policy and original-reference packages")
    if source.manifest.definition != base.manifest.definition:
        raise ValueError("RL original reference differs from the saved policy definition/tokenizer")
    if pin.evidence.tokenizer.kind == "final" and source.manifest.definition.tokenizer != pin.evidence.tokenizer:
        raise ValueError("RL saved tokenizer differs from its immutable inherited pin")
    value = training_publications.metadata(source)
    if str(value.generation.model_id) != (cfg.resume_from or cfg.base_model):
        raise ValueError("RL saved policy metadata differs from its registered model UUID")
    original = training_publications.metadata(base)
    if str(original.generation.model_id) != cfg.base_model:
        raise ValueError("RL original reference metadata differs from its registered base UUID")
    if cfg.resume_from is None:
        return source, base, 0
    mutable = {"model_name", "resume_from", "checkpoint_target"}
    if (value.training_kind != "rl" or value.generation.final or value.generation.total_steps != cfg.iterations
            or value.config.model_dump(exclude=mutable) != cfg.model_dump(exclude=mutable)):
        raise ValueError("RL recovery requires its original request and complete iteration generation")
    return source, base, value.generation.step


def training_group_start(cfg: RlConfig, metadata: TrainingPublication, first: int) -> int:
    """Derive exact next task-seed authority from parent-readable completed state, including inherited curriculum offsets."""
    groups = first * cfg.tasks_per_batch
    if cfg.curriculum is not None:
        saved = metadata.result.get("curriculum")
        if saved is not None:
            if not isinstance(saved, dict):
                raise ValueError("saved RL curriculum metadata is malformed")
            groups = saved.get("training_groups", 0)
        elif cfg.resume_from is not None:
            raise ValueError("RL recovery is missing its completed curriculum metadata")
    if type(groups) is not int or not 0 <= groups < 2**64:
        raise ValueError("saved RL training-group position is invalid")
    begin = cfg.seed_base + groups
    end = begin + (cfg.iterations - first) * cfg.tasks_per_batch
    if end > 2**64 or begin <= cfg.holdout_base < end:
        raise ValueError("effective RL recovery seed range exceeds uint64 or overlaps the queued holdout")
    return groups


def completed_result(result: dict, context: SupervisedRun, prepared: PreparedRl, final: Checkpoint | None) -> dict:
    """Require exact completed schedule/source/output identity and durable registered final bytes after verified cleanup."""
    cfg, definition = prepared.config, prepared.saved.manifest.definition
    if (final is None or not final.final or final.model_id != cfg.model_name or final.step != cfg.iterations
            or final.total_steps != cfg.iterations
            or result.get("model_id") != cfg.model_name or type(result.get("iterations")) is not int
            or result["iterations"] != cfg.iterations or result.get("definition_id") != prepared.saved.manifest.definition_id
            or result.get("base_model") != cfg.base_model or result.get("ctx_window") != definition.context_window
            or result.get("vocab") != definition.tokenizer.vocabulary_size):
        raise ValueError("RL completion differs from its registered final output or inherited request")
    with store.session() as db:
        model = q.get_model(db, UUID(cfg.model_name))
        if model is None or model.definition_id != result["definition_id"] or model.run_id != context.job_id:
            raise ValueError("RL completion has no matching durable model registration")
    return result


class RlHandler:
    """Own one pinned queue job's complete saved-input/effect/session/publication lifecycle under its existing device lease."""

    def run_supervised(self, context: SupervisedRun) -> dict:
        """Admit actual saved packages and durable effects, then retire known gym resources only after child/callback cleanup."""
        cfg = RlConfig.model_validate(context.config)
        validate_configuration(cfg)
        if context.actor is None:
            raise ValueError("supervised RL requires an initiating actor UUID")
        actor = UUID(context.actor)
        with store.session() as db:
            pin = training_jobs.inherited_pin(db, context.job_id)
            run = q.lock_run(db, context.job_id)
            if (pin is None or run is None or run.kind != RunKind.rl or run.training is None
                    or run.training.training_type != "rl" or run.owner_id != actor
                    or run.status not in (RunStatus.running, RunStatus.cancelling)
                    or RlConfig.model_validate(run.config) != cfg):
                raise ValueError("supervised RL requires the actual admitted pinned job and initiating actor")
        check_cancelled(context.cancel)
        prepared = self.prepare(context, cfg, pin)
        effects, admission = prepared.effects, prepared.admission
        rl_admission.reserve_root(storage.gym_effects_dir(), effects)
        directory = storage.gym_effect_dir(context.job_id)
        directory.mkdir(exist_ok=False)
        authority = Authority(job_id=context.job_id, actor_id=actor,
            configuration_sha256=hashlib.sha256(encode_json(cfg.model_dump(mode="json"))).hexdigest(),
            first_iteration=prepared.first_iteration, training_group_start=prepared.training_group_start)
        log = GymEffectLog(directory / "ownership.jsonl", authority, max_bytes=effects.ownership_bytes)
        dispatcher = GymDispatcher(cfg, log, response_bytes=rl_admission.RESPONSE_BYTES)
        broker = EffectBroker(str(actor), dispatcher, EffectJournal(directory / "issued.jsonl", max_bytes=effects.issued_bytes))
        try:
            result = self.execute(context, prepared, broker)
            if (log.pending is not None or any(not terminal for terminal in log.runs.values())
                    or any(not episode.terminal for episode in log.episodes.values())
                    or log.creations["train"] != cfg.iterations - prepared.first_iteration):
                raise ValueError("RL worker completed without its exact settled iteration gym resources")
        except CleanupUncertain:
            raise
        except BaseException:
            broker.close(timeout=admission.policy.deadlines.effect_seconds)
            dispatcher.retire_resources()
            rl_recovery.discard_settled(log)
            raise
        broker.close(timeout=admission.policy.deadlines.effect_seconds)
        dispatcher.retire_resources()
        rl_recovery.discard_settled(log)
        check_cancelled(context.cancel)
        return result

    def prepare(self, context: SupervisedRun, cfg: RlConfig, pin: catalog.PreparedTrainingPin) -> PreparedRl:
        """Freeze registered source/reference bytes and reserve parent evidence memory/disk before copying inputs or starting a worker."""
        source, base, first = source_packages(cfg, pin)
        groups = training_group_start(cfg, training_publications.metadata(source), first)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        snapshot = training_jobs.frozen_validation(pin, cfg, source.manifest.definition.tokenizer, context.job_id, device)
        effects = rl_admission.effect_budget(cfg)
        root = storage.package_staging_dir()
        root.mkdir(parents=True, exist_ok=True)
        available = training_admission.capacity(root)
        available = training_admission.Capacity(available.host_bytes,
            min(available.disk_bytes, shutil.disk_usage(context.workers.root).free))
        available = rl_admission.worker_capacity(available, effects)
        reference = base if cfg.resume_from is not None else None
        admission = training_admission.admit_files((source.directory / "tokenizer.json",),
            (source, reference) if reference is not None else (source,), available, device, reserved_disk_bytes=effects.disk_bytes)
        return PreparedRl(cfg, snapshot, source, reference, first, groups, admission, effects)

    def execute(self, context: SupervisedRun, prepared: PreparedRl, broker: EffectBroker) -> dict:
        """Run one complete RL command with private package/tokenizer grants and the existing acknowledged publication owner."""
        publications = TrainingPublications(context, prepared.admission.storage.package_budget())
        with tempfile.TemporaryDirectory(prefix="rl-inputs-", dir=storage.package_staging_dir()) as temporary:
            directory = Path(temporary)
            source = package_transfer.write_archive(prepared.saved.directory, directory / "policy.tar")
            reference = (package_transfer.write_archive(prepared.reference.directory, directory / "reference.tar")
                         if prepared.reference is not None else None)
            with context.workers.session(context.lease, prepared.admission.policy, "llm.train.rl_worker:RlBackend") as session:
                prep = RlPreparation(request=prepared.config, validation=prepared.snapshot, storage=prepared.admission.storage,
                    tokenizer=session.artifacts.stage_file(prepared.saved.directory / "tokenizer.json", "tokenizer_v1"),
                    reference=session.artifacts.stage_file(reference, "model_package_v1") if reference is not None else None)
                payload = prep.model_dump_json().encode()
                if len(payload) > training_admission.PREPARATION_BYTES:
                    raise ValueError("RL preparation exceeds its admitted config grant")
                command = Train(type="train", job_id=context.job_id, training_kind="rl",
                    config=session.artifacts.write_bytes(payload, "training_config_v1"),
                    recovery=session.artifacts.stage_file(source, "model_package_v1"))
                result = session.run(command, request_id=context.job_id, effects=broker, adopt=publications, cancel=context.cancel)
                if result.outcome != "ok" or result.result is None:
                    raise RequestCancelled("RL worker did not complete")
                require_format(result.result, "training_result_v1")
                value = decode_json(session.artifacts.read_bytes(result.result))
                if not isinstance(value, dict):
                    raise ValueError("RL result must be a bounded JSON summary")
        check_cancelled(context.cancel)
        return completed_result(value, context, prepared, publications.final)
