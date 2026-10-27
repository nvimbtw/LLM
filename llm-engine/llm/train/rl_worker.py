"""Supervised-worker RL policy/reference execution; every gym effect is authorized and issued by the parent."""

from dataclasses import dataclass
import random
import resource
import time

import numpy as np
from tokenizers import Tokenizer
import torch

from llm.architecture.package_integrity import decode_json, encode_json
from llm.architecture.spec import canonical_digest
from llm.gym.effect_client import EffectClient
from llm.model import checkpoint
from llm.orchestrate import practice, rl
from llm.runtime.protocol import Completed, RuntimePins, Train
from llm.runtime.worker import WorkerContext
from llm.train import memory
from llm.train.curriculum import Lessons
from llm.train.model_policy import optimizer_groups
from llm.train.modular_preflight import preflight
from llm.train.modular_worker import Cancellation, TrainingBackend
from llm.train.rl_packages import RlCheckpointWriter, RlPackages
from llm.train.rl_records import RlPreparation


@dataclass
class IterationState:
    """Worker-owned finite rollout state, using explicit definition facts instead of Transformer attributes."""

    tokenizer: Tokenizer
    started: float
    rng: random.Random
    generator: torch.Generator
    window: int
    lessons: Lessons | None = None
    rope: None = None


class RlBackend(TrainingBackend):
    """Own policy/reference/import/optimizer lifetimes with the shared training worker reporting and cleanup contract."""

    def handle(self, command, context: WorkerContext) -> Completed:
        """Validate saved inputs, preflight both models and complete RL within one admitted empty worker session."""
        if (not isinstance(command, Train) or command.training_kind != "rl" or command.pins is not None
                or command.corpus is not None or command.job_id != context.request_id):
            raise ValueError("RL backend requires one inherited empty-session RL command for this request")
        prep = RlPreparation.model_validate_json(encode_json(decode_json(context.artifacts.read_bytes(command.config))))
        if prep.validation.run_id != command.job_id or prep.validation.request.profile.device != context.device:
            raise ValueError("RL preparation differs from its admitted request/device")
        self.phase("load")
        raw = context.artifacts.read_bytes(prep.tokenizer)
        packages = RlPackages(command, prep, raw, context, self.phase)
        random.seed(prep.request.seed_base)
        np.random.seed(prep.request.seed_base % 2**32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        resources = context.bootstrap.policy.resources
        allocator = (min(resources.device_allocator_bytes, memory.budget_bytes("cuda")) if context.device == "cuda"
                     else resources.tree_resident_bytes - baseline)
        self.ready = preflight(prep.validation, allocator_bytes=allocator, progress=self.phase,
            restore=packages.restore, restore_reference=packages.restore_reference,
            recompute=False, recover_exact=prep.request.resume_from is not None)
        receipt = self.ready.validation.report.receipt
        context.pins = RuntimePins(definition_id=canonical_digest(self.ready.definition),
            bindings_digest=receipt.identity.parameter_digest, profile_digest=canonical_digest(receipt.identity.profile),
            receipt_id=receipt.receipt_id)
        self.ready.model.eval()
        with EffectClient(context) as client:
            result = self.learn(prep, packages, Tokenizer.from_str(raw.decode("utf-8")), client)
        artifact = context.artifacts.write_bytes(encode_json(result), "training_result_v1")
        return Completed(type="completed", outcome="ok", result=artifact)

    def iteration_state(self, prep: RlPreparation, packages: RlPackages, tokenizer: Tokenizer,
                        client: EffectClient) -> tuple[IterationState, dict]:
        """Restore completed task/sampling/curriculum state without live gym authority.

        RL does not measure complete marker usage. Preserve known producer facts
        but mark their history incomplete so later SFT cannot reset token rows
        already trained by rollout context/actions.
        """
        cfg = prep.request
        state = IterationState(tokenizer, time.time(), random.Random(cfg.seed_base),
            torch.Generator(device=self.context.device), self.ready.definition.context_window)
        resumed = self.ready.state
        metadata = packages.metadata.result
        provenance = dict(resumed.extra.get("provenance", {})) if resumed is not None else {
            "base_model": cfg.base_model,
            **({"dataset": metadata["dataset"]} if "dataset" in metadata else {})}
        if isinstance(provenance.get("dataset"), dict):
            provenance["dataset"] = {**provenance["dataset"], "marker_history_complete": False}
        saved = resumed.extra.get("curriculum") if resumed is not None else metadata.get("curriculum")
        if cfg.curriculum is not None:
            state.lessons = practice.prepare(cfg, client, saved, inherit_base=False)
        if resumed is not None:
            if cfg.curriculum is not None and saved is None:
                raise ValueError("RL recovery is missing its completed curriculum state")
            state.rng.setstate(resumed.extra["task_rng"])
            state.generator.set_state(resumed.extra["sampling_rng"].cpu())
        return state, provenance

    def learn(self, prep: RlPreparation, packages: RlPackages, tokenizer: Tokenizer, client: EffectClient) -> dict:
        """Advance the existing rollout/objective semantics and publish only fully completed optimizer/status/exam boundaries."""
        cfg = prep.request
        model, reference, resumed = self.ready.model, self.ready.reference, self.ready.state
        state, provenance = self.iteration_state(prep, packages, tokenizer, client)
        writer = RlCheckpointWriter(self.ready, prep, packages.tokenizer, self.context, provenance=provenance)
        optimizer = torch.optim.AdamW(optimizer_groups(model, weight_decay=cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.95))
        if resumed is not None:
            optimizer.load_state_dict(resumed.optimizer)
            checkpoint.restore_rng(resumed.rng)
        first = resumed.step if resumed is not None else 0
        cancel = Cancellation(self.context)
        if state.lessons is not None and resumed is None:
            rl._exam(cfg, client, model, state, provenance, self.report, cancel, 0, initial=True)
        history = list(provenance.get("source_history", []))
        last = None
        for iteration in range(first, cfg.iterations):
            self.context.check_cancelled()
            frame = rl._iterate(cfg, client, model, reference, optimizer, state, iteration)
            if state.lessons is not None:
                if (iteration + 1) % cfg.curriculum.check_every == 0:
                    rl._exam(cfg, client, model, state, provenance, self.report, cancel, iteration + 1)
                provenance["curriculum"] = state.lessons.snapshot()
                frame.diag.update(mastery=state.lessons.report(), curriculum_events=list(state.lessons.events))
            history.append({key: frame.diag[key] for key in
                ("step", "source_scores", "task_sequence", "pass_rate", "signal")})
            provenance["source_history"] = history
            writer.provenance = dict(provenance)
            writer.completed_state = {"task_rng": state.rng.getstate(), "sampling_rng": state.generator.get_state(),
                **({"curriculum": state.lessons.snapshot()} if state.lessons is not None else {})}
            self.report(frame)
            rl._checkpoint(cfg, writer, model, optimizer, iteration, frame, self.report)
            last = frame
        self.context.check_cancelled()
        if state.lessons is not None:
            rl._exam(cfg, client, model, state, provenance, self.report, cancel, cfg.iterations, final=True)
        tail = history[-1] if history else {}
        result = {**provenance, "iterations": cfg.iterations, "resumed_at": first,
            "pass_rate": tail.get("pass_rate", 0.0), "signal": tail.get("signal", 0.0),
            "source_scores": tail.get("source_scores", {}), "source_history": history,
            "final_loss": last.loss if last is not None else None, "elapsed_secs": time.time() - state.started,
            "model_id": cfg.model_name, "definition_id": canonical_digest(self.ready.definition),
            "vocab": self.ready.definition.tokenizer.vocabulary_size, "ctx_window": state.window,
            "parameters": self.ready.validation.report.unique_parameter_count}
        writer.provenance = dict(provenance)
        writer.finish(model, result)
        return result
