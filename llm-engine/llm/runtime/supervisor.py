"""Parent process ownership, admission, bounded sessions and restart reconciliation."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import select
import shutil
import socket
import subprocess
import sys
import threading
import time
from uuid import UUID, uuid4

from llm.model.device import Busy, DeviceGate, DeviceLease, GATE
from llm.runtime.artifacts import ArtifactStore
from llm.runtime.bootstrap import WorkerBootstrap
from llm.runtime.effects import EffectBroker
from llm.runtime.processes import (
    OwnershipRecord, identity_is_alive, mark_quarantine, read_record,
    reap_owned_tree, sample_process, write_record,
)
from llm.runtime.protocol import (
    ArtifactAdopted, ArtifactReady, Cancel, CleanupVerified, Completed, EffectReply,
    EffectRequest, Failed, Inspect, RuntimeEnvelope, RuntimePayload, RuntimePolicy, Shutdown,
)
from llm.runtime.transport import receive, send, socket_pair

EventCallback = Callable[[RuntimePayload], None]
ArtifactAdopter = Callable[[ArtifactReady, ArtifactStore], None]


class WorkerStopped(RuntimeError):
    """Execution stopped without a verified successful result; cleanup is checked separately."""


class CleanupUncertain(WorkerStopped):
    """Admission remains quarantined because worker/tree cleanup could not be certified."""


class RequestCancelled(WorkerStopped):
    """The parent cancelled and verified cleanup; durable handlers may mark the job cancelled."""


class WorkerSession:
    """Own one guardian, child-local backend and the actual parent admission capability."""

    def __init__(
        self, manager: WorkerManager, lease: DeviceLease | None, policy: RuntimePolicy,
        backend: str, *, metadata_only: bool = False,
    ) -> None:
        """Register/fsync process ownership before allowing custom backend execution."""
        self.manager = manager
        self.lease = lease
        self.policy = policy
        self.worker_id = uuid4()
        self.directory = manager.root / str(self.worker_id)
        self.bootstrap = WorkerBootstrap(
            worker_id=self.worker_id, lease_id=lease.id if lease else uuid4(), startup_request_id=uuid4(),
            policy=policy, backend=backend, metadata_only=metadata_only,
        )
        self.history: deque[RuntimeEnvelope] = deque(maxlen=policy.retained_progress)
        self._data_sequence = 0
        self._control_sequence = 0
        self._worker_sequence = 0
        self._request_id = self.bootstrap.startup_request_id
        self._closed = False
        self._closing = False
        self._running = False
        self._effects: EffectBroker | None = None
        self._adoption: Future[ArtifactAdopted] | None = None
        self._certificate: CleanupVerified | None = None
        self._lease_scope = None
        self._evict_on_idle = False
        self._request_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._callback_lock = threading.Lock()
        self._control_lock = threading.Lock()
        self._certificate_lock = threading.Lock()
        with ExitStack() as unlaunched:
            self.directory.mkdir(mode=0o700)
            unlaunched.callback(shutil.rmtree, self.directory)
            self.artifacts = ArtifactStore(self.directory / "artifacts", policy.resources)
            self.channel, child_data = socket_pair(policy.pending_messages)
            unlaunched.callback(self.channel.close)
            unlaunched.callback(child_data.close)
            self.control, child_control = socket_pair(4)
            unlaunched.callback(self.control.close)
            unlaunched.callback(child_control.close)
            if lease is not None:
                lease.bind_worker(self.worker_id)
            unlaunched.pop_all()
        try:
            self.guardian = self._launch(child_data, child_control)
        except BaseException:
            self.channel.close()
            self.control.close()
            raise

    def _launch(self, child_data: socket.socket, child_control: socket.socket) -> subprocess.Popen:
        """Own startup handles after binding; uncertain launch retains records and quarantines admission."""
        start_read = start_write = None
        try:
            (self.directory / "bootstrap.json").write_text(self.bootstrap.model_dump_json())
            write_record(self.directory, OwnershipRecord(status="starting"))
            start_read, start_write = os.pipe()
            guardian = subprocess.Popen(
                [sys.executable, "-I", str(Path(__file__).with_name("entry.py")), "guardian", str(self.directory),
                 str(child_data.fileno()), str(child_control.fileno()), str(self.manager.lock_fd), str(start_read)],
                pass_fds=(child_data.fileno(), child_control.fileno(), self.manager.lock_fd, start_read),
                close_fds=True, start_new_session=True, env=worker_environment(self.policy),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            identity = sample_process(guardian.pid).identity
            write_record(self.directory, OwnershipRecord(status="starting", guardian=identity))
            os.write(start_write, b"1")
            return guardian
        except BaseException:
            self.manager._stopping = True
            mark_quarantine(self.manager.root, "guardian launch did not complete verified ownership handoff")
            self.manager.gate.quarantine("guardian launch did not complete verified ownership handoff")
            raise
        finally:
            child_data.close()
            child_control.close()
            if start_read is not None:
                os.close(start_read)
            if start_write is not None:
                os.close(start_write)

    def start(self) -> None:
        """Wait for bounded child-only backend import/construction, without a per-frame reset."""
        deadline = time.monotonic() + self.policy.deadlines.import_seconds + self.policy.deadlines.construction_seconds + 2
        result = self._drain(deadline=deadline)
        if result.outcome != "ok":
            raise WorkerStopped("worker startup was cancelled")

    @property
    def closed(self) -> bool:
        """Expose session retirement to the resident-model owner without allowing it to mutate worker lifecycle state."""
        return self._closed or self._closing

    def _send(self, payload: RuntimePayload) -> None:
        """Send one parent data command; the guardian serializes its separate cancellation channel."""
        self._data_sequence += 1
        send(self.channel, RuntimeEnvelope(
            worker_id=self.worker_id, request_id=self._request_id, sequence=self._data_sequence, payload=payload,
        ))

    def _send_control(self, payload: Cancel | Shutdown) -> None:
        """Send cancellation independently of request callbacks and a full progress/data queue."""
        with self._control_lock:
            self._control_sequence += 1
            self.control.settimeout(0.1)
            send(self.control, RuntimeEnvelope(
                worker_id=self.worker_id, request_id=self._request_id,
                sequence=self._control_sequence, payload=payload,
            ))

    def run(
        self, payload: RuntimePayload, *, request_id: UUID | None = None,
        on_event: EventCallback | None = None, effects: EffectBroker | None = None,
        adopt: ArtifactAdopter | None = None, cancel: threading.Event | None = None,
    ) -> Completed:
        """Execute one full operation, retaining model/optimizer state for subsequent requests."""
        if not self._request_lock.acquire(blocking=False):
            raise WorkerStopped("one request may execute per worker")
        try:
            if self._closed or self._closing:
                raise WorkerStopped("worker session is closed")
            if self.lease is not None:
                self.lease.require_active()
            elif not isinstance(payload, Inspect):
                raise Busy("metadata workers cannot execute numerical model requests")
            self._request_id = request_id or uuid4()
            self._effects = effects
            self._running = True
            self._send(payload)
            return self._drain(on_event=on_event, effects=effects, adopt=adopt, cancel=cancel)
        except BaseException:
            self.close()
            raise
        finally:
            self._running = False
            self._request_lock.release()
            if self._evict_on_idle:
                self.close()

    def _drain(
        self, *, deadline: float | None = None, on_event: EventCallback | None = None,
        effects: EffectBroker | None = None, adopt: ArtifactAdopter | None = None,
        cancel: threading.Event | None = None,
    ) -> Completed:
        """Drain bounded events while guardian watchdogs remain independent from parent effects."""
        pending_effect: Future[EffectReply] | None = None
        pending_adoption: Future[ArtifactAdopted] | None = None
        while not self._closed and not self._closing:
            if deadline is not None and time.monotonic() > deadline:
                raise WorkerStopped("worker startup deadline exceeded")
            if cancel is not None and cancel.is_set():
                self._send_control(Cancel(type="cancel", target_request_id=self._request_id, reason="user"))
                self.close()
                raise RequestCancelled("worker request cancelled")
            if pending_effect is not None and pending_effect.done():
                self._send(pending_effect.result())
                pending_effect = None
            if pending_adoption is not None and pending_adoption.done():
                self._send(pending_adoption.result())
                pending_adoption = None
            readable, _, _ = select.select([self.channel, self.control], [], [], 0.02)
            if self.channel in readable:
                message = self._read_event()
                payload = message.payload
                if isinstance(payload, EffectRequest):
                    if effects is None or pending_effect is not None:
                        raise WorkerStopped("worker effect has no available parent owner")
                    pending_effect = self._begin_effect(effects, payload)
                elif isinstance(payload, ArtifactReady):
                    if pending_adoption is not None:
                        raise WorkerStopped("worker has multiple pending publications")
                    pending_adoption = self._adopt(payload, adopt)
                elif isinstance(payload, Failed):
                    raise WorkerStopped(f"{payload.code}: {payload.message}")
                elif isinstance(payload, Completed):
                    if pending_effect is not None or pending_adoption is not None:
                        raise WorkerStopped("worker completed before parent acknowledgement")
                    if payload.result is not None:
                        self.artifacts.verify(payload.result)
                    if on_event is not None:
                        on_event(payload)
                    return payload
                if on_event is not None:
                    on_event(payload)
            if self.control in readable:
                self._receive_certificate()
                raise WorkerStopped("worker terminated before successful completion")
        raise WorkerStopped("worker session closed during execution")

    def _read_event(self) -> RuntimeEnvelope:
        """Authenticate a worker event and retain only a bounded recent report history."""
        try:
            message = receive(self.channel)
        except (OSError, EOFError) as error:
            raise WorkerStopped("worker exited without a successful result") from error
        message.require_sender("worker")
        message.require_context(
            worker_id=self.worker_id, request_id=self._request_id, next_sequence=self._worker_sequence + 1,
        )
        self._worker_sequence = message.sequence
        self.history.append(message)
        return message

    def _adopt(self, notice: ArtifactReady, adopter: ArtifactAdopter | None) -> Future[ArtifactAdopted]:
        """Own a parent adoption callback before launching it; closure atomically prevents new publication work."""
        with self._callback_lock:
            if adopter is None or self._closing:
                raise WorkerStopped("worker publication has no available parent adopter")
            future: Future[ArtifactAdopted] = Future()
            self._adoption = future
            threading.Thread(
                target=self._adopt_callback, args=(notice, adopter, future),
                name="llm-runtime-publication", daemon=True,
            ).start()
            return future

    def _begin_effect(self, effects: EffectBroker, request: EffectRequest) -> Future[EffectReply]:
        """Serialize external-effect issuance with session closure while the broker journals its write identity."""
        with self._callback_lock:
            if self._closing:
                raise WorkerStopped("worker effect session is closing")
            return effects.begin(request, self.artifacts)

    def _drain_parent_callbacks(self) -> None:
        """Wait for actual effect/adoption completion before grant retirement; completed adoption errors do not imply live resources."""
        if self._effects is not None:
            self._effects.close(timeout=self.policy.deadlines.effect_seconds)
        if self._adoption is not None:
            self._adoption.exception(timeout=self.policy.deadlines.artifact_adoption_seconds)

    def _adopt_callback(
        self, notice: ArtifactReady, adopter: ArtifactAdopter, future: Future[ArtifactAdopted],
    ) -> None:
        """Verify bytes, finish parent publication and return its explicit acknowledgement once."""
        try:
            self.artifacts.verify(notice.artifact)
            adopter(notice, self.artifacts)
            future.set_result(ArtifactAdopted(
                type="artifact_adopted", publication_id=notice.publication_id,
                artifact_id=notice.artifact.artifact_id, outcome="adopted",
            ))
        except Exception as error:
            future.set_exception(error)

    def _receive_certificate(self) -> None:
        """Accept cleanup only from the guardian channel for this exact worker and lease."""
        with self._certificate_lock:
            if self._certificate is not None:
                return
            message = receive(self.control)
            message.require_sender("guardian")
            if message.worker_id != self.worker_id or not isinstance(message.payload, CleanupVerified):
                raise CleanupUncertain("runtime cleanup certificate identity mismatch")
            message.payload.require_lease(self.bootstrap.lease_id)
            self._certificate = message.payload

    def close(self) -> None:
        """Reap the worker tree and drain parent callbacks before retiring grants or releasing admission; uncertainty quarantines both."""
        with self._close_lock:
            if self._closed:
                return
            with self._callback_lock:
                self._closing = True
            clean = False
            try:
                if self._certificate is None:
                    try:
                        self._send_control(Shutdown(type="shutdown", reason="shutdown"))
                    except OSError:
                        pass
                    deadline = time.monotonic() + self.policy.deadlines.cancel_grace_seconds + self.policy.deadlines.reap_seconds + 1
                    while self._certificate is None and time.monotonic() < deadline:
                        if select.select([self.control], [], [], 0.02)[0]:
                            self._receive_certificate()
                self.guardian.wait(timeout=self.policy.deadlines.reap_seconds)
                self._drain_parent_callbacks()
                clean = self._certificate is not None and self._certificate.outcome == "reaped"
            except (OSError, EOFError, ValueError, TimeoutError, subprocess.TimeoutExpired):
                clean = False
            finally:
                self._closed = True
                self.channel.close()
                self.control.close()
                self.manager._retire(self, clean)
                if self._lease_scope is not None:
                    self._lease_scope.__exit__(None, None, None)
                    self._lease_scope = None
            if not clean:
                raise CleanupUncertain("worker cleanup is uncertain; admission remains closed")

    def __enter__(self) -> WorkerSession:
        """Use an already started session under an explicit cleanup-owning context."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop/reap this worker when its request/job/cache lifetime ends."""
        self.close()


class WorkerManager:
    """Own one engine lock, session registry and crash/restart cleanup barrier."""

    def __init__(self, root: Path, gate: DeviceGate = GATE) -> None:
        """Prepare process-local orchestration; start() owns the lock before queue admission."""
        self.root = root
        self.gate = gate
        self.lock_fd = -1
        self._sessions: set[WorkerSession] = set()
        self._lock = threading.RLock()
        self._stopping = False
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None

    def start(self, timeout: float = 5.0) -> None:
        """Acquire a stable cross-process lock and reconcile old ownership before execution."""
        self.gate.require_parent()
        if self.lock_fd >= 0:
            raise RuntimeError("runtime manager is already started")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(self.root / "engine.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        deadline = time.monotonic() + timeout
        try:
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise Busy("another engine or guardian owns runtime execution")
                    time.sleep(0.02)
            self.lock_fd = descriptor
            if (self.root / "quarantine.json").exists():
                raise CleanupUncertain("runtime root is quarantined after uncertain cleanup")
            self._reconcile()
            self._monitor_thread = threading.Thread(
                target=self._monitor_residents, name="llm-runtime-residents", daemon=True,
            )
            self._monitor_thread.start()
        except BaseException:
            self.lock_fd = -1
            os.close(descriptor)
            raise

    def _monitor_residents(self) -> None:
        """Retire idle serving workers after guardian expiry so their device leases become available.

        Poll healthy residents without borrowing request ownership. Only a ready
        guardian channel needs the request lock to keep retirement separate from
        execution; durable job handlers continue to own their cleanup.
        """
        while not self._monitor_stop.wait(0.05):
            with self._lock:
                sessions = tuple(self._sessions)
            for session in sessions:
                if session._lease_scope is None or session._closed:
                    continue
                try:
                    if not select.select([session.control], [], [], 0)[0]:
                        continue
                    if not session._request_lock.acquire(blocking=False):
                        continue
                    try:
                        if not session._closed and select.select([session.control], [], [], 0)[0]:
                            session.close()
                    finally:
                        session._request_lock.release()
                except CleanupUncertain:
                    pass
                except (OSError, ValueError):
                    if not session._closed:
                        mark_quarantine(self.root, "resident worker monitoring failed")
                        self.gate.quarantine("resident worker monitoring failed")

    def _reconcile(self) -> None:
        """Verify/stop recorded prior workers without ever killing a reused unrelated PID."""
        for directory in self.root.iterdir():
            if not directory.is_dir():
                continue
            try:
                UUID(directory.name)
            except ValueError:
                continue
            if not (directory / "ownership.json").exists():
                shutil.rmtree(directory)
                continue
            record = read_record(directory)
            if record.status == "uncertain":
                raise CleanupUncertain("prior runtime cleanup was uncertain")
            if record.status != "clean":
                if record.guardian is not None and identity_is_alive(record.guardian):
                    raise CleanupUncertain("prior guardian remains alive without a verified lock handoff")
                if record.worker is not None and not reap_owned_tree(record.worker, 2.0, adopt=False):
                    raise CleanupUncertain("prior worker tree could not be reaped")
            shutil.rmtree(directory)

    def session(
        self, lease: DeviceLease | None, policy: RuntimePolicy, backend: str, *, metadata_only: bool = False,
    ) -> WorkerSession:
        """Start one usable session on an existing lease, or a CPU-only metadata worker without one."""
        with self._lock:
            if self.lock_fd < 0 or self._stopping:
                raise WorkerStopped("runtime manager is not accepting sessions")
            if len(self._sessions) >= 8:
                raise WorkerStopped("runtime session count budget exceeded")
            if lease is None and (not metadata_only or policy.resources.device != "cpu"):
                raise Busy("numerical workers require the actual parent device lease")
            if lease is not None and lease.gate is not self.gate:
                raise Busy("worker lease belongs to another device admission owner")
            if policy.resources.tree_memory_enforcement != "preflight_and_monitor":
                raise WorkerStopped("no delegated cgroup was configured; refusing an unenforced hard tree policy")
            session = WorkerSession(self, lease, policy, backend, metadata_only=metadata_only)
            self._sessions.add(session)
        try:
            session.start()
            return session
        except BaseException:
            session.close()
            raise

    def resident_session(self, policy: RuntimePolicy, backend: str) -> WorkerSession:
        """Own a serving-cache lease across requests, released by eviction/idle/shutdown cleanup."""
        scope = self.gate.admit()
        lease = scope.__enter__()
        try:
            session = self.session(lease, policy, backend)
            session._lease_scope = scope
            return session
        except BaseException:
            scope.__exit__(None, None, None)
            raise

    def evict_for_queue(self) -> None:
        """Give queued work priority by evicting idle serving caches after active requests finish."""
        with self._lock:
            sessions = [session for session in self._sessions if session._lease_scope is not None]
        for session in sessions:
            session._evict_on_idle = True
            if not session._running:
                session.close()

    def _retire(self, session: WorkerSession, clean: bool) -> None:
        """Release a verified worker binding or persist quarantine before registry removal."""
        if clean:
            if session.lease is not None:
                session.lease.finish_worker(session.worker_id)
            shutil.rmtree(session.directory)
        else:
            mark_quarantine(self.root, "worker cleanup could not be verified")
            self.gate.quarantine("worker cleanup could not be verified")
        with self._lock:
            self._sessions.discard(session)

    def close(self) -> None:
        """Stop/drain all session owners before releasing the never-unlinked engine lock file."""
        self._monitor_stop.set()
        with self._lock:
            self._stopping = True
            sessions = tuple(self._sessions)
        errors = []
        for session in sessions:
            try:
                session.close()
            except CleanupUncertain as error:
                errors.append(error)
        if self._monitor_thread is not None:
            self._monitor_thread.join()
            self._monitor_thread = None
        if self.lock_fd >= 0:
            os.close(self.lock_fd)
            self.lock_fd = -1
        if errors:
            raise CleanupUncertain("one or more workers could not be stopped") from errors[0]

    def __enter__(self) -> WorkerManager:
        """Start the engine runtime ownership barrier for a bounded caller lifetime."""
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        """Drain sessions before the manager context releases engine ownership."""
        self.close()


def worker_environment(policy: RuntimePolicy) -> dict[str, str]:
    """Allow only runtime tuning values; never inherit browser, service or storage credentials."""
    environment = {
        "PATH": os.defpath, "LANG": "C.UTF-8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1", "MAX_JOBS": "1", "LLM_RUNTIME_CHILD": "1",
    }
    if policy.resources.device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment
