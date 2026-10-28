"""Linux process ownership, crash records and bounded owned-tree resource accounting."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import signal
import time
from typing import Literal
from uuid import UUID

from pydantic import Field

from llm.architecture.spec import Contract
from llm.runtime.artifacts import sync_directory

BOOT_ID = UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip())


class ProcessIdentity(Contract):
    """Kernel birth identity preventing a recycled PID from authorizing termination."""

    pid: int = Field(strict=True, gt=0)
    start_ticks: int = Field(strict=True, ge=0)
    boot_id: UUID


class OwnershipRecord(Contract):
    """Fsynced startup barrier record; custom execution starts only after worker identity publication."""

    status: Literal["starting", "active", "clean", "uncertain"]
    guardian: ProcessIdentity | None = None
    worker: ProcessIdentity | None = None


class ProcessSample(Contract):
    """A process's group/session and resident bytes at one observable kernel snapshot."""

    identity: ProcessIdentity
    parent_pid: int
    group_id: int
    session_id: int
    resident_bytes: int
    state: str


def sample_process(pid: int) -> ProcessSample:
    """Read Linux stat fields without confusing spaces/parentheses in process names."""
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return ProcessSample(
        identity=ProcessIdentity(pid=pid, start_ticks=int(fields[19]), boot_id=BOOT_ID),
        parent_pid=int(fields[1]), group_id=int(fields[2]), session_id=int(fields[3]),
        resident_bytes=max(0, int(fields[21])) * os.sysconf("SC_PAGE_SIZE"), state=fields[0],
    )


def identity_is_alive(identity: ProcessIdentity) -> bool:
    """Check the recorded birth identity; never treat PID reuse as a surviving owned process."""
    try:
        return sample_process(identity.pid).identity == identity
    except (FileNotFoundError, ProcessLookupError):
        return False


def group_processes(worker: ProcessIdentity) -> list[ProcessSample]:
    """Enumerate the owned session/group and fail on ambiguous live leader PID reuse.

    Ordinary helpers inherit the worker session; arbitrary session detachment and
    credential changes are outside the supported administrator SDK contract.
    """
    try:
        current = sample_process(worker.pid)
        if current.identity != worker:
            raise ValueError("runtime_pid_reused")
    except (FileNotFoundError, ProcessLookupError):
        pass
    result = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            process = sample_process(int(directory.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            if directory.stat().st_uid == os.getuid():
                raise
            continue
        if process.group_id == worker.pid and process.session_id == worker.pid:
            if process.identity.start_ticks < worker.start_ticks:
                raise ValueError("runtime_process_ownership_uncertain")
            result.append(process)
    return result


def terminate_group(worker: ProcessIdentity) -> None:
    """Send SIGKILL only after checking recorded group ownership against PID reuse."""
    if not group_processes(worker):
        return
    try:
        os.killpg(worker.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def configure_subreaper() -> None:
    """Make a guardian adopt/reap ordinary orphaned helpers instead of leaking zombies."""
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")


def require_guardian(parent_pid: int) -> None:
    """Bind worker death to its guardian, closing the prctl/parent-death startup race."""
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
    if os.getppid() != parent_pid:
        raise RuntimeError("runtime_guardian_already_exited")


def write_record(directory: Path, record: OwnershipRecord) -> None:
    """Durably replace ownership evidence before releasing either executable-start barrier."""
    target = directory / "ownership.json"
    pending = directory / "ownership.pending"
    with pending.open("w") as output:
        output.write(record.model_dump_json())
        output.flush()
        os.fsync(output.fileno())
    os.replace(pending, target)
    sync_directory(directory)


def read_record(directory: Path) -> OwnershipRecord:
    """Read bounded trusted crash metadata without deserializing executable Python state."""
    path = directory / "ownership.json"
    if path.stat().st_size > 4096:
        raise ValueError("runtime_ownership_record_budget")
    return OwnershipRecord.model_validate_json(path.read_bytes())


def reap_owned_tree(worker: ProcessIdentity, timeout: float, *, adopt: bool) -> bool:
    """Terminate/reap owned descendants within a fixed deadline; uncertainty returns false."""
    terminate_group(worker)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if adopt:
            while True:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    break
        if not group_processes(worker):
            return True
        time.sleep(0.01)
    return False


def mark_quarantine(root: Path, reason: str) -> None:
    """Persist a bounded failure marker so restart cannot forget uncertain cleanup."""
    path = root / "quarantine.json"
    with path.open("w") as output:
        json.dump({"reason": reason[:1024]}, output)
        output.flush()
        os.fsync(output.fileno())
    sync_directory(root)


def tree_disk_usage(directory: Path, byte_limit: int) -> int:
    """Bound all session output/cache files and entry count without following symlinks."""
    total = 0
    for count, path in enumerate(directory.rglob("*"), 1):
        if count > 4096 or path.is_symlink():
            raise ValueError("runtime_output_entries")
        if path.is_file():
            total += path.stat().st_size
            if total > byte_limit:
                raise ValueError("runtime_output_bytes")
    return total
