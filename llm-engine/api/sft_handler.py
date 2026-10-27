"""Saved-definition supervised SFT preparation under the existing queue, lease and publication owners."""

from pathlib import Path
import tempfile

import store
import torch

from api import sft_authority, training_admission, training_jobs
from api.training_handler import PreparedSftRequest, PreparedTraining, TrainingHandler, check_cancelled, saved_package
from api.worker import SupervisedRun
from llm import storage
from llm.architecture import package
from llm.architecture.package_integrity import read_bytes
from llm.architecture.tokenizer_binding import TOKENIZER_BYTES
from llm.config import SftConfig, TrainConfig
from llm.corpus import special
from llm.orchestrate import modular_sft_data
from llm.runtime.supervisor import RequestCancelled
from llm.train.model_preflight import sft_loop_config
from llm.train.sft_tokenizer import prepare_tokenizer
from store import architecture_queries as catalog, queries as q
from store.schema import RunStatus


def original_base(cfg: SftConfig, saved: package.VerifiedPackage,
                  pin: catalog.PreparedTrainingPin) -> package.VerifiedPackage:
    """Verify the registered original reference and compatible tokenizer bytes without loading tensors or source."""
    if cfg.resume_from is None:
        return saved
    base = saved_package(TrainConfig(model_name=cfg.model_name, resume_from=cfg.base_model), pin)
    source = saved.manifest.definition
    raw = read_bytes(base.directory / "tokenizer.json", TOKENIZER_BYTES)
    used = [special.SPECIAL_TOKENS.index(marker) for marker in (special.MEMORY_OPEN, special.MEMORY_CLOSE)]
    target = read_bytes(saved.directory / "tokenizer.json", TOKENIZER_BYTES)
    if source.tokenizer == base.manifest.definition.tokenizer:
        expected, binding = prepare_tokenizer(raw, base.manifest.definition.tokenizer, [])
    else:
        expected, binding = prepare_tokenizer(raw, base.manifest.definition.tokenizer, used)
    if source != base.manifest.definition.model_copy(update={"tokenizer": binding}) or target != expected:
        raise ValueError("SFT original base differs from the recovered definition or compatible tokenizer bytes")
    return base


class SftHandler(TrainingHandler):
    """Prepare authorized private SFT data and delegate the numerical session to the existing training adapter."""

    def run_supervised(self, context: SupervisedRun) -> dict:
        """Freeze saved inputs and derived binding before numerical admission; retain authority for interrupted adoption."""
        cfg = SftConfig.model_validate(context.config)
        with store.session() as db:
            pin = training_jobs.inherited_pin(db, context.job_id)
            run = q.lock_run(db, context.job_id)
            if (pin is None or run is None or run.training is None or run.training.training_type != "sft"
                    or run.status not in (RunStatus.running, RunStatus.cancelling)):
                raise ValueError("supervised SFT requires the actual admitted pinned job")
        _recipe, window, _names = training_jobs.frozen_recipe(pin)
        effective = sft_loop_config(cfg, window)
        training_admission.validate_configuration(effective)
        check_cancelled(context.cancel)
        saved = saved_package(TrainConfig(model_name=cfg.model_name, resume_from=cfg.resume_from or cfg.base_model,
            recover_exact=cfg.resume_from is not None), pin)
        base = original_base(cfg, saved, pin)
        inherited = (saved.manifest.definition.tokenizer, base.manifest.definition.tokenizer) if cfg.resume_from else (saved.manifest.definition.tokenizer,)
        if pin.evidence.tokenizer.kind == "final" and pin.evidence.tokenizer not in inherited:
            raise ValueError("SFT saved tokenizer differs from the inherited enqueue pin")
        root = storage.package_staging_dir()
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sft-inputs-", dir=root) as temporary:
            directory = Path(temporary)
            budget = training_admission.packing_budget(training_admission.capacity(directory))
            try:
                prepared = modular_sft_data.prepare(cfg, saved=saved, base=base, directory=directory,
                    actor=context.actor, progress=context.progress, cancel=context.cancel, budget=budget)
            except InterruptedError as error:
                if context.cancel.is_set():
                    raise RequestCancelled("SFT cancelled during parent data preparation") from error
                raise
            check_cancelled(context.cancel)
            authority = sft_authority.prepare(context.job_id, cfg, saved, prepared)
            sft_authority.publish(authority)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            snapshot = training_jobs.frozen_validation(pin, effective, prepared.data.binding, context.job_id, device,
                prepared_tokenizer=authority.tokenizer)
            available = training_admission.capacity(directory)
            runtime_disk = training_admission.capacity(context.workers.root).disk_bytes
            available = training_admission.Capacity(available.host_bytes, min(available.disk_bytes, runtime_disk))
            reference = base if cfg.resume_from is not None and prepared.chat_probe is not None else None
            admission = training_admission.admit(prepared.data, saved, available, device,
                additional_inputs=prepared.additional_paths, reference=reference)
            request = PreparedTraining(effective, snapshot, prepared.data, saved, admission,
                PreparedSftRequest(cfg, prepared, reference))
            result = self.execute(context, request, directory)
        check_cancelled(context.cancel)
        sft_authority.discard(context.job_id)
        return result
