"""Independent trusted process-tree owner, cancellation watchdog and cleanup certifier."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import time

from llm.runtime.bootstrap import WorkerBootstrap, read_bootstrap
from llm.runtime.processes import (
    OwnershipRecord, ProcessIdentity, configure_subreaper, group_processes,
    mark_quarantine, read_record, reap_owned_tree, sample_process, tree_disk_usage, write_record,
)
from llm.runtime.protocol import Cancel, CleanupVerified, Failed, RuntimeEnvelope, Shutdown
from llm.runtime.transport import receive, socket_pair
from llm.runtime.watchdog import Watchdog


@dataclass
class Guardian:
    """Own all channels, bounded forwarding and monotonic wire identities for one worker."""

    directory: Path
    bootstrap: WorkerBootstrap
    parent: socket.socket
    control: socket.socket
    worker: socket.socket
    identity: ProcessIdentity
    watchdog: Watchdog
    parent_sequence: int = 0
    worker_sequence: int = 0
    control_sequence: int = 0
    forwarded_sequence: int = 0
    stopping_at: float | None = None
    failure_code: str | None = None
    to_parent: deque[bytes] = field(default_factory=deque)
    to_worker: deque[bytes] = field(default_factory=deque)

    def queue(self, queue: deque[bytes], message: RuntimeEnvelope) -> None:
        """Bound queued IPC even when either consumer stops draining progress."""
        if len(queue) >= self.bootstrap.policy.pending_messages:
            raise ValueError("runtime_progress_queue_budget")
        queue.append(message.to_packet())

    def forward(self, message: RuntimeEnvelope) -> None:
        """Serialize control and data commands into one validated worker-side sequence."""
        self.forwarded_sequence += 1
        self.queue(self.to_worker, message.model_copy(update={"sequence": self.forwarded_sequence}))

    def read_parent(self, now: float) -> None:
        """Admit one parent command or matching acknowledgement from the data channel."""
        message = receive(self.parent)
        message.require_sender("parent")
        message.require_context(
            worker_id=self.bootstrap.worker_id, request_id=message.request_id,
            next_sequence=self.parent_sequence + 1,
        )
        self.parent_sequence = message.sequence
        if isinstance(message.payload, (Cancel, Shutdown)):
            raise ValueError("runtime_control_on_data_channel")
        self.watchdog.command(message, now)
        self.forward(message)

    def read_control(self, now: float) -> None:
        """Handle cancellation/liveness independently from data forwarding and parent callbacks."""
        message = receive(self.control)
        message.require_sender("parent")
        message.require_context(
            worker_id=self.bootstrap.worker_id, request_id=message.request_id,
            next_sequence=self.control_sequence + 1,
        )
        self.control_sequence = message.sequence
        if not isinstance(message.payload, (Cancel, Shutdown)):
            raise ValueError("runtime_invalid_control")
        if isinstance(message.payload, Cancel) and message.payload.target_request_id != self.watchdog.request_id:
            raise ValueError("runtime_cancel_wrong_request")
        if self.stopping_at is None:
            self.stopping_at = now + self.bootstrap.policy.deadlines.cancel_grace_seconds
        self.forward(message)

    def read_worker(self, now: float) -> None:
        """Validate each event before allowing it to update a deadline or reach the parent."""
        message = receive(self.worker)
        message.require_sender("worker")
        message.require_context(
            worker_id=self.bootstrap.worker_id, request_id=self.watchdog.request_id,
            next_sequence=self.worker_sequence + 1,
        )
        self.worker_sequence = message.sequence
        self.watchdog.event(message, now)
        self.queue(self.to_parent, message)

    def monitor(self, now: float) -> None:
        """Enforce observable helper/RSS/output bounds and independent request deadlines."""
        resources = self.bootstrap.policy.resources
        samples = group_processes(self.identity)
        if len(samples) > resources.owned_processes:
            raise ValueError("runtime_process_count_budget")
        if sum(process.resident_bytes for process in samples) > resources.tree_resident_bytes:
            raise ValueError("runtime_tree_memory_budget")
        tree_disk_usage(self.directory, resources.total_artifact_bytes + 65536)
        if self.watchdog.expired(now) and self.stopping_at is None:
            self.failure_code = "runtime_deadline_exceeded"
            self.stopping_at = now + self.bootstrap.policy.deadlines.cancel_grace_seconds
            cancel = Cancel(type="cancel", target_request_id=self.watchdog.request_id, reason="deadline")
            self.forward(RuntimeEnvelope(
                worker_id=self.bootstrap.worker_id, request_id=self.watchdog.request_id,
                sequence=0, payload=cancel,
            ))

    def loop(self) -> None:
        """Relay bounded traffic until cooperative stop, policy failure or parent death."""
        next_monitor = 0.0
        while self.stopping_at is None or time.monotonic() < self.stopping_at:
            writable = ([self.parent] if self.to_parent else []) + ([self.worker] if self.to_worker else [])
            readable, ready, _ = select.select([self.control, self.parent, self.worker], writable, [], 0.01)
            now = time.monotonic()
            if self.control in readable:
                self.read_control(now)
            if self.parent in readable:
                self.read_parent(now)
            if self.worker in readable:
                self.read_worker(now)
            for channel, queue in ((self.parent, self.to_parent), (self.worker, self.to_worker)):
                if channel in ready:
                    channel.send(queue.popleft())
            if now >= next_monitor:
                self.monitor(now)
                next_monitor = now + self.bootstrap.policy.resources.monitor_interval_seconds


def spawn_worker(directory: Path, bootstrap: WorkerBootstrap) -> tuple[subprocess.Popen, socket.socket, ProcessIdentity]:
    """Fsync child birth ownership before permitting any backend import or custom execution."""
    channel, child = socket_pair(bootstrap.policy.pending_messages)
    start_read, start_write = os.pipe()
    try:
        worker = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).with_name("entry.py")), "worker", str(directory),
             str(child.fileno()), str(start_read), str(os.getpid())],
            pass_fds=(child.fileno(), start_read), close_fds=True, start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        identity = sample_process(worker.pid).identity
        record = read_record(directory)
        write_record(directory, record.model_copy(update={"worker": identity, "status": "active"}))
        os.write(start_write, b"1")
        return worker, channel, identity
    finally:
        child.close()
        os.close(start_read)
        os.close(start_write)


def certify_cleanup(guardian: Guardian) -> bool:
    """Kill/reap the recorded tree and persist uncertainty before releasing the engine lock."""
    try:
        clean = reap_owned_tree(guardian.identity, guardian.bootstrap.policy.deadlines.reap_seconds, adopt=True)
    except (OSError, ValueError):
        clean = False
    record = read_record(guardian.directory)
    write_record(guardian.directory, record.model_copy(update={"status": "clean" if clean else "uncertain"}))
    if not clean:
        mark_quarantine(guardian.directory.parent, "guardian could not verify worker tree cleanup")
    if guardian.failure_code is not None:
        send_failure(guardian)
    certificate = CleanupVerified(
        type="cleanup_verified", lease_id=guardian.bootstrap.lease_id,
        outcome="reaped" if clean else "uncertain", owned_processes_remaining=0 if clean else 1,
    )
    try:
        guardian.control.settimeout(0.1)
        guardian.control.sendall(RuntimeEnvelope(
            worker_id=guardian.bootstrap.worker_id, request_id=guardian.watchdog.request_id,
            sequence=1, payload=certificate,
        ).to_packet())
    except OSError:
        pass
    return clean


def send_failure(guardian: Guardian) -> None:
    """Forward a bounded trusted watchdog error without letting a stalled consumer delay cleanup."""
    message = RuntimeEnvelope(
        worker_id=guardian.bootstrap.worker_id, request_id=guardian.watchdog.request_id,
        sequence=guardian.worker_sequence + 1,
        payload=Failed(type="failed", phase="cleanup", code=guardian.failure_code, message="Worker stopped by runtime supervision"),
    )
    try:
        guardian.parent.settimeout(0.05)
        for packet in guardian.to_parent:
            guardian.parent.sendall(packet)
        guardian.parent.sendall(message.to_packet())
    except OSError:
        pass


def run_guardian(directory: Path, data_fd: int, control_fd: int, lock_fd: int) -> None:
    """Own the shared engine lock until tree cleanup; never import a custom backend."""
    configure_subreaper()
    bootstrap = read_bootstrap(directory)
    parent, control = socket.socket(fileno=data_fd), socket.socket(fileno=control_fd)
    _, worker, identity = spawn_worker(directory, bootstrap)
    for channel in (parent, control, worker):
        channel.setblocking(False)
    watchdog = Watchdog(bootstrap.policy.deadlines, bootstrap.startup_request_id)
    watchdog.bootstrap(time.monotonic())
    guardian = Guardian(directory, bootstrap, parent, control, worker, identity, watchdog)
    try:
        guardian.loop()
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    except (OSError, ValueError) as error:
        code = str(error)
        guardian.failure_code = code if code.startswith("runtime_") and len(code) <= 64 else "runtime_transport_failed"
    finally:
        certify_cleanup(guardian)
        for channel in (parent, control, worker):
            channel.close()
        os.close(lock_fd)
