"""Parent-owned host/disk admission for authorized preparation and supervised pretraining copies."""

from dataclasses import dataclass
from pathlib import Path
import shutil
import math

from llm.architecture import package, package_transfer
from llm.architecture.package_integrity import PackageBudget
from llm.corpus.pack import PackBudget
from llm.corpus import special
from llm.config import TrainConfig
from llm.orchestrate.modular_data import MAX_CORPUS_BYTES, MAX_DOCUMENTS, PreparedData
from llm.runtime.protocol import RuntimeDeadlines, RuntimePolicy, RuntimeResources
from llm.train import memory
from llm.train.modular_records import TrainingStorage

DISK_RESERVE_BYTES = 512 << 20
MAX_HOST_BYTES = 8 << 30
MIN_HOST_BYTES = 2 << 30
PREPARATION_BYTES = 8 << 20


def validate_configuration(cfg: TrainConfig) -> None:
    """Reject unsupported input cardinality, invalid training schedules and nonfinite optimizer values before acquisition."""
    stages = cfg.stages()
    if (not 1 <= len(stages) <= 128 or sum(len(stage.datasets) for stage in stages) > 4096
            or any(stage.epochs < 1 for stage in stages)):
        raise ValueError("training requires positive epochs and bounded stages/dataset selections")
    if cfg.batch_size < 1 or not 0 <= cfg.seed < 2**64 or cfg.warmup_steps < 0:
        raise ValueError("training requires a positive batch, uint64 seed and nonnegative warmup")
    if any(not math.isfinite(value) or value < 0 for value in (cfg.lr, cfg.weight_decay, cfg.grad_clip)):
        raise ValueError("training optimizer values must be finite and nonnegative")
    minimum = len(special.token_list()) + 256
    if cfg.resume_from is None and cfg.tokenizer_id is None and not minimum <= cfg.max_vocab <= 65536:
        raise ValueError(f"derived byte-level training vocabulary must contain {minimum} to 65536 entries")


@dataclass(frozen=True)
class Capacity:
    """Available host memory and free disk bytes observed under the existing parent admission owner."""

    host_bytes: int
    disk_bytes: int


@dataclass(frozen=True)
class Admission:
    """One coherent runtime tree policy and exact per-generation package ceilings."""

    policy: RuntimePolicy
    storage: TrainingStorage


def available_host_bytes() -> int:
    """Read Linux available memory, tightening it by a finite enclosing unified-cgroup allowance when present."""
    with Path("/proc/meminfo").open() as source:
        fields = source.read(65536).splitlines()
    available = next((int(line.split()[1]) * 1024 for line in fields if line.startswith("MemAvailable:")), None)
    if available is None:
        raise ValueError("training host-memory admission is unavailable")
    ceiling = Path("/sys/fs/cgroup/memory.max")
    current = Path("/sys/fs/cgroup/memory.current")
    if ceiling.is_file() and current.is_file():
        limit = ceiling.read_text().strip()
        if limit != "max":
            available = min(available, max(0, int(limit) - int(current.read_text())))
    return available


def capacity(directory: Path) -> Capacity:
    """Reserve host headroom and observe the actual staging filesystem before acquiring or copying more data."""
    host = min(MAX_HOST_BYTES, available_host_bytes() * 3 // 4)
    if host < MIN_HOST_BYTES:
        raise ValueError("training requires at least 2 GiB of available host admission after headroom")
    return Capacity(host_bytes=host, disk_bytes=shutil.disk_usage(directory).free)


def packing_budget(available: Capacity) -> PackBudget:
    """Keep acquisition output and retained span/index arrays within the observed disk/host allowance."""
    corpus = min(MAX_CORPUS_BYTES, (available.disk_bytes - DISK_RESERVE_BYTES) // 6)
    documents = min(MAX_DOCUMENTS, available.host_bytes // 256)
    if corpus < 1 << 20 or documents < 1:
        raise ValueError("insufficient capacity for training corpus preparation")
    return PackBudget(max_corpus_bytes=corpus, max_documents=documents)


def admit(prepared: PreparedData, saved: package.VerifiedPackage | None, available: Capacity, device: str,
          *, additional_inputs: tuple[Path, ...] = (), reference: package.VerifiedPackage | None = None) -> Admission:
    """Budget inputs/recovery copies and six concurrent generation copies before creating a numerical session.

    New corpus cache bytes already occupy disk when this snapshot is taken.
    Recovery needs a parent archive, private grant and worker materialization.
    A generation may coexist as raw files, sealed package, archive, grant,
    parent staging and the last retained complete generation. Current capacity
    is rechecked during publication; prior explicit keeps never authorize growth.
    """
    paths = (prepared.corpus, prepared.spans, prepared.tokenizer, *additional_inputs)
    packages = tuple(value for value in (saved, reference) if value is not None)
    return admit_files(paths, packages, available, device)


def admit_files(paths: tuple[Path, ...], packages: tuple[package.VerifiedPackage, ...], available: Capacity, device: str,
                *, reserved_disk_bytes: int = 0) -> Admission:
    """Admit exact input files, private saved-package copies and six generations while reserving parent-owned durable effect storage.

    Packed-data training and gym-based RL share serialization/copy accounting;
    their domain preparation and resource ledgers remain separately owned.
    Additional parent storage is subtracted before any session or archive copy.
    """
    if type(reserved_disk_bytes) is not int or reserved_disk_bytes < 0 or not paths:
        raise ValueError("training admission requires actual inputs and a nonnegative parent disk reservation")
    input_sizes = tuple(path.stat().st_size for path in paths)
    inputs = sum(input_sizes) + PREPARATION_BYTES
    recovery = 0
    package_sizes = []
    for checked in packages:
        payload = sum(value.size_bytes for value in checked.integrity.files.values()) + (4 << 20)
        package_sizes.append(package_transfer.archive_limit(PackageBudget(total_bytes=payload, file_bytes=min(payload, 16 << 30))))
    recovery = sum(package_sizes)
    generation = min(32 << 30, (available.disk_bytes - inputs - 3 * recovery - reserved_disk_bytes - DISK_RESERVE_BYTES) // 6)
    if generation < 8 << 20:
        raise ValueError("insufficient disk admission for training package/recovery copies")
    storage = TrainingStorage(package_bytes=generation, file_bytes=min(16 << 30, generation))
    archive = package_transfer.archive_limit(storage.package_budget())
    allocator = memory.budget_bytes(device) if device == "cuda" else 0
    resources = RuntimeResources(process_address_bytes=1 << 40 if device == "cuda" else 64 << 30,
        tree_resident_bytes=available.host_bytes, tree_memory_enforcement="preflight_and_monitor",
        device=device, device_allocator_bytes=allocator, owned_processes=2,
        artifact_bytes=max(*input_sizes, recovery, archive, PREPARATION_BYTES),
        total_artifact_bytes=inputs + 2 * recovery + 4 * archive + PREPARATION_BYTES)
    return Admission(storage=storage, policy=RuntimePolicy(resources=resources,
        deadlines=RuntimeDeadlines(validation_seconds=180.0, artifact_adoption_seconds=120.0)))
