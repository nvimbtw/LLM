"""Shared exact input/runtime declarations for supervised language-model preflight."""

import torch
from typing import Protocol

from llm.architecture.spec import RuntimeProfile, ShapeDomain
from llm.config import RlConfig, SftConfig, TrainConfig


def sft_loop_config(cfg: SftConfig, context_window: int) -> TrainConfig:
    """Translate SFT optimization/execution options to the shared loop without supplying recipe structure.

    Context comes from the saved definition. Fresh SFT uses saved weights with
    a new optimizer; a resume requests exact masked-data cursor recovery.
    """
    fields = ("model_name", "datasets", "epochs", "lr", "batch_size", "warmup_steps", "grad_clip", "weight_decay",
              "shuffle", "seed", "recompute_blocks", "checkpoint_target", "rebuild_dataset",
              "precision", "compile_model", "allow_tf32")
    return TrainConfig(**{name: getattr(cfg, name) for name in fields}, ctx_window=context_window,
                       resume_from=cfg.resume_from or cfg.base_model, recover_exact=cfg.resume_from is not None)


class TrainingProfileInputs(Protocol):
    """Share explicit numerical input controls between queued training and data-only selection preview."""

    batch_size: int
    precision: str
    seed: int
    allow_tf32: bool


def training_profile(cfg: TrainingProfileInputs, context_window: int, device: str) -> RuntimeProfile:
    """Declare full-prefix train/eval evidence through the configured batch, including one-row serving and scoring."""
    if device not in {"cpu", "cuda"} or cfg.batch_size < 1 or context_window < 1 or cfg.seed < 0:
        raise ValueError("invalid training preflight device, batch, context or numerical seed")
    precision = "fp32" if device == "cpu" else cfg.precision
    if device == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        precision = "fp16"
    return RuntimeProfile(torch_version=str(torch.__version__), sdk_version=1, device=device,
        precision=precision, capture_policy="torch-export-strict-v1",
        domains=(ShapeDomain(batch_min=1, batch_max=cfg.batch_size,
                            sequence_min=1, sequence_max=context_window, signatures=("full_prefix",)),),
        modes=("train", "eval"), allow_tf32=cfg.allow_tf32 if device == "cuda" else False,
        numerical_seed=cfg.seed)


def rl_profile(cfg: RlConfig, context_window: int, device: str) -> RuntimeProfile:
    """Declare full-prefix fp32 policy/reference evidence for actual rollout and exam batches without changing legacy arithmetic."""
    if device not in {"cpu", "cuda"} or context_window < 1 or not 0 <= cfg.seed_base < 2**64:
        raise ValueError("invalid RL preflight device, inherited context or numerical seed")
    batch = cfg.tasks_per_batch * cfg.group_size
    if cfg.curriculum is not None:
        batch = max(batch, min(cfg.micro_batch, max(cfg.curriculum.validation_samples, cfg.curriculum.test_samples)))
    return RuntimeProfile(torch_version=str(torch.__version__), sdk_version=1, device=device,
        precision="fp32", capture_policy="torch-export-strict-v1",
        domains=(ShapeDomain(batch_min=1, batch_max=batch, sequence_min=1,
                            sequence_max=context_window, signatures=("full_prefix",)),),
        modes=("train", "eval"), allow_tf32=False, numerical_seed=cfg.seed_base)
