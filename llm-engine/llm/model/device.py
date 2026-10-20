"""Parent-owned device admission shared by legacy work and supervised sessions."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from threading import Condition
from uuid import UUID, uuid4


class Busy(ValueError):
    """Device admission is occupied, queued, unavailable or quarantined."""


@dataclass(frozen=True)
class DeviceLease:
    """One parent-created device lifetime, never serialized or passed to a worker."""

    id: UUID
    gate: DeviceGate
    owner_pid: int

    def require_active(self) -> None:
        """Refuse use after release, in a child, or following uncertain cleanup."""
        self.gate.require_active(self)

    def quarantine(self, reason: str) -> None:
        """Keep admission closed when this lifetime cannot prove stopped execution."""
        self.gate.quarantine(reason)

    def bind_worker(self, worker_id: UUID) -> None:
        """Bind the actual worker lifetime so exiting a lease cannot release a live child."""
        self.gate.bind_worker(self, worker_id)

    def finish_worker(self, worker_id: UUID) -> None:
        """Remove a worker binding only after its parent has verified cleanup."""
        self.gate.finish_worker(self, worker_id)


class DeviceGate:
    """Own one numerical lease, queue priority and permanent cleanup quarantine."""

    def __init__(self) -> None:
        """Create a process-bound admission owner with no independent worker-side copy."""
        self._owner_pid = os.getpid()
        self._condition = Condition()
        self._active: DeviceLease | None = None
        self._worker: UUID | None = None
        self._waiters = 0
        self._quarantined: str | None = None

    def require_parent(self) -> None:
        """Deny inherited/fresh worker-side gates even when a model imports this module."""
        if os.getpid() != self._owner_pid or os.environ.get("LLM_RUNTIME_CHILD") == "1":
            raise Busy("Model workers cannot acquire parent device admission")

    @contextmanager
    def admit(self, *, wait: bool = False):
        """Yield a single lease, preserving queued-work priority and quarantine on exit."""
        self.require_parent()
        with self._condition:
            if not wait and (self._active is not None or self._waiters):
                raise Busy("The device is busy; retry after the active execution finishes")
            self._waiters += 1
            try:
                while self._active is not None and self._quarantined is None:
                    if not wait:
                        raise Busy("The device is busy")
                    self._condition.wait()
                if self._quarantined is not None:
                    raise Busy("Device admission is quarantined after uncertain worker cleanup")
                lease = DeviceLease(uuid4(), self, self._owner_pid)
                self._active = lease
            finally:
                self._waiters -= 1
        try:
            yield lease
        finally:
            uncertain = False
            with self._condition:
                if self._active is lease and self._quarantined is None:
                    if self._worker is not None:
                        self._quarantined = "device lease exited with an unverified live worker"
                        uncertain = True
                    else:
                        self._active = None
                self._condition.notify_all()
            if uncertain:
                raise Busy("Device lease exited before worker cleanup was verified")

    def require_active(self, lease: DeviceLease) -> None:
        """Validate an explicit capability against the actual current parent lease."""
        self.require_parent()
        with self._condition:
            if self._active is not lease or self._quarantined is not None:
                raise Busy("Device lease is inactive or quarantined")

    def quarantine(self, reason: str) -> None:
        """Permanently close admission and wake waiters to fail instead of waiting forever."""
        self.require_parent()
        with self._condition:
            self._quarantined = reason[:1024]
            self._condition.notify_all()

    def bind_worker(self, lease: DeviceLease, worker_id: UUID) -> None:
        """Enforce exactly one worker for an admitted device lifetime."""
        self.require_active(lease)
        with self._condition:
            if self._worker is not None:
                raise Busy("The device lease already owns a model worker")
            self._worker = worker_id

    def finish_worker(self, lease: DeviceLease, worker_id: UUID) -> None:
        """Verify the parent's worker identity before allowing normal lease release."""
        self.require_parent()
        with self._condition:
            if self._active is not lease or self._worker != worker_id:
                raise Busy("Worker does not own this device lease")
            self._worker = None


GATE = DeviceGate()


@contextmanager
def admit(*, wait: bool = False):
    """Yield the application's sole device lease; legacy callers may ignore the value."""
    with GATE.admit(wait=wait) as lease:
        yield lease
