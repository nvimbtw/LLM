"""Typed inherited RL preparation and completed-iteration input/runtime boundaries."""

import math
from typing import Literal
from uuid import UUID

from pydantic import model_validator

from llm.architecture.spec import Contract
from llm.architecture.validation_records import ValidationSnapshot
from llm.config import RlConfig
from llm.runtime.protocol import RuntimeArtifact, require_format
from llm.train.model_preflight import rl_profile
from llm.train.modular_records import TrainingStorage

MAX_RL_HISTORY_BYTES = 4 << 20


def validate_configuration(cfg: RlConfig) -> None:
    """Bound effective schedules, identifiers and finite optimization/sampling values before numerical construction or gym acquisition."""
    for value in (cfg.model_name, cfg.base_model, cfg.resume_from):
        if value is not None:
            UUID(value)
    if (not 1 <= cfg.tasks_per_batch <= 256 or not 2 <= cfg.group_size <= 256
            or not 1 <= cfg.max_rounds <= 32 or cfg.micro_batch < 1 or cfg.iterations < 1
            or cfg.tasks_per_batch * cfg.group_size > 256 or not 1 <= len(cfg.task_types) <= 4096
            or not 1 <= len(cfg.gym_id) <= 128 or any(not 1 <= len(entry.id) <= 128 for entry in cfg.task_types)
            or len(cfg.gym_weights) > 4096 or any(not 1 <= len(key) <= 128 for key in cfg.gym_weights)
            or not 0 <= cfg.seed_base < 2**64 or not 0 <= cfg.holdout_base < 2**64
            or cfg.seed_base + cfg.iterations * cfg.tasks_per_batch > 2**64 or cfg.iterations >= 2**53):
        raise ValueError("RL requires bounded gym batches, task identifiers, uint64 seeds and an exact iteration schedule")
    if any(not math.isfinite(value) or value < 0 for value in (cfg.lr, cfg.beta, cfg.grad_clip, cfg.weight_decay)):
        raise ValueError("RL optimizer and reference penalty values must be finite and nonnegative")
    if (not math.isfinite(cfg.temperature) or cfg.temperature <= 0 or not math.isfinite(cfg.top_p)
            or not 0 < cfg.top_p <= 1 or cfg.top_k < 0 or not 1 <= cfg.max_tokens <= 65536):
        raise ValueError("RL sampling requires finite positive temperature, valid top-p/top-k and bounded tokens")
    if cfg.context_multiplier != 1.0:
        raise ValueError("declared RL context extension requires explicit capability/profile evidence")
    sources = min(128, len(cfg.task_types), cfg.tasks_per_batch) if cfg.gym_id == "general" else 1
    history = cfg.iterations * (2048 + 128 * cfg.tasks_per_batch + 512 * sources)
    if cfg.curriculum is not None:
        history += len(cfg.task_types) * (4096 + 1024 * (cfg.iterations // cfg.curriculum.check_every + 2))
    if history > MAX_RL_HISTORY_BYTES:
        raise ValueError("RL schedule exceeds the admitted checkpoint/history metadata allowance")


class RlPreparation(Contract):
    """Immutable saved-definition preflight, tokenizer and original-reference grants for one complete RL request."""

    schema_version: Literal[1] = 1
    request: RlConfig
    validation: ValidationSnapshot
    tokenizer: RuntimeArtifact
    reference: RuntimeArtifact | None = None
    storage: TrainingStorage = TrainingStorage()

    @model_validator(mode="after")
    def final_bindings(self):
        """Require exact inherited full-prefix evidence and original reference recovery before any source execution."""
        validate_configuration(self.request)
        require_format(self.tokenizer, "tokenizer_v1")
        if self.reference is not None:
            require_format(self.reference, "model_package_v1")
        if self.request.resume_from is not None and self.reference is None:
            raise ValueError("RL recovery requires the original frozen base package")
        request = self.validation.request
        if (self.validation.recipe is None or request.mode != "full" or request.subject.kind != "revision"
                or request.subject.reference.kind != "architecture" or request.profile is None
                or request.context_window is None or request.tokenizer is None or request.tokenizer.kind != "final"):
            raise ValueError("RL requires a final inherited causal revision snapshot")
        if (self.tokenizer.sha256 != request.tokenizer.fingerprint
                or request.profile != rl_profile(self.request, request.context_window, request.profile.device)):
            raise ValueError("RL tokenizer or input/runtime profile differs from the inherited request")
        return self
