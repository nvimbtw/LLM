"""Finite parent RL ledger admission alongside the shared numerical package/copy policy."""

from dataclasses import dataclass
from pathlib import Path
import stat
from uuid import UUID

from api import training_admission
from llm.architecture.package_integrity import encode_json
from llm.config import RlConfig
from llm.train.rl_records import validate_configuration

MAX_LOG_BYTES = 64 << 20
MAX_EFFECT_ROOT_BYTES = 512 << 20
MAX_RETAINED_JOBS = 1024
RESPONSE_BYTES = 8 << 20


@dataclass(frozen=True)
class EffectBudget:
    """Hard domain/operation journal ceilings and conservative parent memory reserved before worker admission."""

    ownership_bytes: int
    issued_bytes: int
    parent_memory_bytes: int

    @property
    def disk_bytes(self) -> int:
        """Return both durable ledgers' combined on-disk reservation in bytes."""
        return self.ownership_bytes + self.issued_bytes


def effect_budget(cfg: RlConfig) -> EffectBudget:
    """Reserve schedule-based evidence allowance with exact write ceilings; unusual service metadata can still exhaust its explicit quota."""
    validate_configuration(cfg)
    runs = cfg.iterations
    episodes = runs * cfg.tasks_per_batch * cfg.group_size
    if cfg.curriculum is not None:
        validations = len(cfg.task_types) * (cfg.iterations // cfg.curriculum.check_every + 2)
        tests = 2 * len(cfg.task_types)
        runs += validations + tests
        episodes += validations * cfg.curriculum.validation_samples + tests * cfg.curriculum.test_samples
    writes = 2 * runs + episodes * (cfg.max_rounds + 1)
    configuration = len(encode_json(cfg.model_dump(mode="json")))
    ownership = max(16 << 20, (1 << 20) + writes * 2048 + episodes * 1024 + runs * configuration)
    issued = max(1 << 20, writes * 64)
    if ownership > MAX_LOG_BYTES or issued > MAX_LOG_BYTES:
        raise ValueError("RL schedule exceeds the admitted durable gym-effect allowance")
    return EffectBudget(ownership, issued, 16 * ownership + 4 * RESPONSE_BYTES)


@dataclass(frozen=True)
class EvidenceInventory:
    """Bounded existing job directories and actual retained evidence bytes for admission or exclusive startup cleanup."""

    jobs: tuple[Path, ...]
    size_bytes: int


def inventory(root: Path) -> EvidenceInventory:
    """Inspect at most the retained job/file/byte limits, rejecting unexpected or aliased storage without loading journal bodies."""
    if not root.exists():
        return EvidenceInventory((), 0)
    jobs = []
    used = 0
    for index, directory in enumerate(root.iterdir()):
        if index >= MAX_RETAINED_JOBS:
            raise ValueError("RL retained gym-effect job allowance is exhausted")
        if directory.is_symlink() or not directory.is_dir() or str(UUID(directory.name)) != directory.name:
            raise ValueError("RL gym-effect root contains an unexpected entry")
        jobs.append(directory)
        for count, path in enumerate(directory.iterdir()):
            if count >= 2 or path.name not in {"ownership.jsonl", "issued.jsonl"}:
                raise ValueError("RL gym-effect job contains unexpected evidence")
            value = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(value.st_mode):
                raise ValueError("RL gym-effect evidence must be a regular owned file")
            used += value.st_size
            if used > MAX_EFFECT_ROOT_BYTES:
                raise ValueError("RL retained gym-effect disk allowance is exhausted")
    return EvidenceInventory(tuple(jobs), used)


def reserve_root(root: Path, budget: EffectBudget) -> None:
    """Reserve another ledger's hard ceilings under the exclusive queue owner, keeping aggregate retained jobs and disk bounded."""
    root.mkdir(parents=True, exist_ok=True)
    retained = inventory(root)
    if len(retained.jobs) >= MAX_RETAINED_JOBS or retained.size_bytes + budget.disk_bytes > MAX_EFFECT_ROOT_BYTES:
        raise ValueError("RL retained gym-effect admission is exhausted")


def worker_capacity(available: training_admission.Capacity, budget: EffectBudget) -> training_admission.Capacity:
    """Reserve bounded parent ledger/response memory instead of assigning it to the numerical worker tree too."""
    host = available.host_bytes - budget.parent_memory_bytes
    if host < 1 << 30:
        raise ValueError("RL has insufficient host admission after parent effect/response reservation")
    return training_admission.Capacity(host, available.disk_bytes)
