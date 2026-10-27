"""Real production worker lifecycle, admission, resource and crash/restart evidence."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from uuid import UUID, uuid4

import pytest

from llm.model.device import Busy, DeviceGate
from llm.runtime.effects import EffectBroker, EffectJournal
from llm.runtime.processes import (
    group_processes, read_record, reap_owned_tree, sample_process, terminate_group,
)
from llm.runtime.protocol import (
    Generate, Load, RuntimeDeadlines, RuntimePins, RuntimePolicy, RuntimeResources, Train,
)
from llm.runtime.supervisor import CleanupUncertain, WorkerManager, WorkerStopped
from llm.runtime import supervisor, transport


class SetupEndpoint:
    """Record transport setup and closure while injecting a specific socket option failure."""

    def __init__(self, failure: int) -> None:
        """Fail on the selected option call; zero leaves this endpoint successful."""
        self.failure = failure
        self.calls = 0
        self.closed = False

    def setsockopt(self, *values) -> None:
        """Count option setup and raise at the configured denied operation."""
        self.calls += 1
        if self.calls == self.failure:
            raise PermissionError("denied socket configuration")

    def settimeout(self, seconds: float) -> None:
        """Accept the timeout only after successful option configuration."""
        assert seconds == 1.0

    def close(self) -> None:
        """Record release of an endpoint created before setup failed."""
        self.closed = True


@pytest.mark.parametrize("failure", [1, 2, 3, 4])
def test_denied_socket_setup_closes_both_endpoints(monkeypatch, failure):
    """Every partial option-setup failure closes both newly allocated handles."""
    endpoints = (SetupEndpoint(failure if failure <= 2 else 0), SetupEndpoint(failure - 2))
    monkeypatch.setattr(transport.socket, "socketpair", lambda *args: endpoints)
    with pytest.raises(PermissionError, match="denied socket"):
        transport.socket_pair()
    assert all(endpoint.closed for endpoint in endpoints)


@pytest.mark.parametrize("failure", [1, 2])
def test_unlaunched_channel_failure_removes_directory_and_releases_admission(runtime, monkeypatch, failure):
    """A prelaunch failure leaves no directory or handles and never binds a nonexistent worker."""
    manager, gate = runtime
    endpoints = []
    calls = 0

    def channels(pending):
        """Create unconfigured local handles until the requested setup boundary fails."""
        nonlocal calls
        calls += 1
        if calls == failure:
            raise PermissionError("denied worker channels")
        pair = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        endpoints.extend(pair)
        return pair

    monkeypatch.setattr(supervisor, "socket_pair", channels)
    with gate.admit() as lease:
        with pytest.raises(PermissionError, match="denied worker channels"):
            manager.session(lease, policy(), "tests.runtime_backend:ResidentBackend")
    assert all(endpoint.fileno() == -1 for endpoint in endpoints)
    assert not any(path.is_dir() for path in manager.root.iterdir())
    assert not manager._sessions
    with gate.admit():
        pass


def test_failed_guardian_handoff_closes_handles_and_persists_quarantine(runtime, monkeypatch):
    """Launch uncertainty keeps durable records, closes local handles and blocks every new session."""
    manager, gate = runtime
    endpoints = []

    def channels(pending):
        """Provide real local handles without the sandbox-denied buffer configuration."""
        pair = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        endpoints.extend(pair)
        return pair

    def failed_launch(*args, **kwargs):
        """Represent an uncertain OS launch boundary without creating a process."""
        raise OSError("guardian handoff interrupted")

    monkeypatch.setattr(supervisor, "socket_pair", channels)
    monkeypatch.setattr(supervisor.subprocess, "Popen", failed_launch)
    with gate.admit() as lease:
        with pytest.raises(OSError, match="handoff interrupted"):
            manager.session(lease, policy(), "tests.runtime_backend:ResidentBackend")
    assert all(endpoint.fileno() == -1 for endpoint in endpoints)
    assert (manager.root / "quarantine.json").is_file()
    assert any(path.is_dir() for path in manager.root.iterdir())
    with pytest.raises(Busy, match="quarantined"):
        with gate.admit():
            pass
    with pytest.raises(WorkerStopped, match="not accepting"):
        manager.session(None, policy(), "tests.runtime_backend:ResidentBackend", metadata_only=True)


def policy(**deadlines) -> RuntimePolicy:
    """Return a conservative CPU policy with short explicit failure-test watchdogs."""
    return RuntimePolicy(
        deadlines=RuntimeDeadlines(
            **{"import_seconds": 10.0, "construction_seconds": 10.0,
               "training_progress_seconds": 0.4, "inference_seconds": 0.4, **deadlines},
        ),
        resources=RuntimeResources(
            process_address_bytes=16 << 30, tree_resident_bytes=2 << 30,
            tree_memory_enforcement="preflight_and_monitor", device="cpu", device_allocator_bytes=0,
            owned_processes=2, artifact_bytes=8 << 20, total_artifact_bytes=16 << 20,
        ),
    )


@pytest.fixture
def runtime(tmp_path):
    """Own and drain a disposable manager with an independent real parent admission gate."""
    gate = DeviceGate()
    with WorkerManager(tmp_path / "runtime", gate) as manager:
        yield manager, gate


def load(session) -> RuntimePins:
    """Bind fixture model identity after a successful finite worker-side forward."""
    pins = RuntimePins(
        definition_id="a" * 64, bindings_digest="b" * 64,
        profile_digest="c" * 64, receipt_id=uuid4(),
    )
    definition = session.artifacts.write_bytes(b"{}", "definition_v1")
    assert session.run(Load(type="load", pins=pins, definition=definition)).outcome == "ok"
    return pins


def training(session, pins, **config) -> Train:
    """Stage one whole-job configuration instead of sending tensor-operation RPCs."""
    artifact = session.artifacts.write_bytes(json.dumps({"steps": 3, **config}).encode(), "training_config_v1")
    return Train(type="train", pins=pins, job_id=uuid4(), training_kind="pretrain", config=artifact)


@pytest.mark.parametrize("operation", ["train", "generate"])
@pytest.mark.parametrize("backend", ["DomainFailure", "ContractFailure"])
def test_training_domain_reason_is_bounded_and_serving_reason_remains_private(runtime, operation, backend):
    """Real worker packets retain actionable private training codes while public generation gets only the exception class."""
    manager, gate = runtime
    with gate.admit() as lease:
        session = manager.session(lease, policy(), f"tests.runtime_backend:{backend}")
        pins = load(session)
        command = training(session, pins) if operation == "train" else Generate(type="generate", pins=pins,
            prompt=session.artifacts.write_bytes(b"{}", "prompt_v1"), seed=0, temperature=0.0, max_tokens=1)
        with pytest.raises(WorkerStopped):
            session.run(command)
        event = next(event for event in session.history if event.payload.type == "failed")
        if operation == "train" and backend == "DomainFailure":
            assert event.payload.code == "memory_estimate"
            assert event.payload.message.startswith("unsupported captured operation:")
            assert len(event.payload.message) == 512
        elif operation == "train":
            assert event.payload.code == "runtime_backend_failed"
            assert event.payload.message.startswith("ValueError: saved input contract differs:")
            assert len(event.payload.message) == 512
        else:
            assert event.payload.code == "runtime_backend_failed"
            assert event.payload.message == ("ArchitectureError" if backend == "DomainFailure" else "ValueError")
        assert len(event.to_packet()) <= 4096
        assert session.closed


def generation(session, pins, tokens=5) -> Generate:
    """Stage an explicitly bounded request whose cache stays inside the resident backend."""
    prompt = session.artifacts.write_bytes(b'{"tokens":[1,2]}', "prompt_v1")
    return Generate(type="generate", pins=pins, prompt=prompt, max_tokens=tokens, seed=52, temperature=0.8)


def wait_gone(pid: int, timeout: float = 5) -> bool:
    """Wait for actual process reaping rather than treating a stopped/zombie PID as cleanup."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            return True
        time.sleep(0.02)
    return False


def test_resident_training_optimizer_and_serving_share_one_process(runtime):
    """Multiple full training jobs and generations retain the model/AdamW until explicit cleanup."""
    manager, gate = runtime
    with gate.admit() as lease:
        with manager.session(lease, policy(), "tests.runtime_backend:ResidentBackend") as session:
            pins = load(session)
            first = session.run(training(session, pins))
            before = json.loads(session.artifacts.read_bytes(first.result))
            second = session.run(training(session, pins, steps=1))
            after = json.loads(session.artifacts.read_bytes(second.result))
            assert (before["steps"], after["steps"]) == (3, 4)
            assert before["pid"] == after["pid"]
            assert session.run(generation(session, pins)).outcome == "ok"
            time.sleep(0.6)
            assert session.run(generation(session, pins, tokens=2)).outcome == "ok"
            with pytest.raises(Busy):
                with gate.admit():
                    pass
        assert wait_gone(before["pid"])
    with gate.admit():
        pass


@pytest.mark.parametrize("backend,limits", [
    ("tests.runtime_import_hang:Backend", {"import_seconds": 0.2}),
    ("tests.runtime_backend:ConstructionHang", {"construction_seconds": 0.2}),
])
def test_import_and_constructor_hangs_are_stopped(runtime, backend, limits):
    """Independent guardian watchdogs bound actual source import and constructor execution."""
    manager, gate = runtime
    with gate.admit() as lease:
        with pytest.raises(WorkerStopped):
            manager.session(lease, policy(**limits), backend)
    with gate.admit():
        pass


@pytest.mark.parametrize("backend", ["ForwardHang", "BackwardHang", "Crash", "Allocation", "Output"])
def test_validated_runtime_failures_stop_and_release_the_real_gate(runtime, backend):
    """Forward/autograd hangs, crashes, large allocations and stdout never survive their owning lease."""
    manager, gate = runtime
    with gate.admit() as lease:
        session = manager.session(lease, policy(), f"tests.runtime_backend:{backend}")
        pins = load(session)
        identity = read_record(session.directory).worker
        with pytest.raises(WorkerStopped):
            session.run(training(session, pins))
        assert wait_gone(identity.pid)
    with gate.admit():
        pass


def test_late_decode_keeps_the_last_adopted_checkpoint(runtime, tmp_path):
    """Forced decoding termination preserves the complete checkpoint already adopted by the parent."""
    manager, gate = runtime
    published = tmp_path / "published.pt"

    def adopt(notice, artifacts):
        """Represent durable parent publication by copying verified complete bytes outside the session."""
        published.write_bytes(artifacts.read_bytes(notice.artifact))

    with gate.admit() as lease:
        session = manager.session(lease, policy(), "tests.runtime_backend:DecodeHang")
        pins = load(session)
        session.run(training(session, pins, checkpoint=True), adopt=adopt)
        complete = published.read_bytes()
        with pytest.raises(WorkerStopped):
            session.run(generation(session, pins))
        assert published.read_bytes() == complete
        assert max(event.payload.completed for event in session.history if event.payload.type == "progress" and event.payload.phase == "generation") == 3


def test_legitimate_training_uses_progress_not_short_validation_deadline(runtime):
    """A complete training operation may outlive validation while actual completed batches advance."""
    manager, gate = runtime
    limits = policy(validation_seconds=0.1, training_progress_seconds=0.35)
    with gate.admit() as lease, manager.session(lease, limits, "tests.runtime_backend:ResidentBackend") as session:
        pins = load(session)
        started = time.monotonic()
        session.run(training(session, pins, steps=6, delay=0.1))
        assert time.monotonic() - started > limits.deadlines.validation_seconds


def test_real_cancel_does_not_report_success_before_reaping(runtime):
    """A user cancellation reaches a cooperative boundary, then the guardian fully reaps the worker."""
    manager, gate = runtime
    cancel = threading.Event()

    def report(event):
        """Cancel after an actual optimizer update rather than before admission."""
        if event.type == "progress" and event.unit == "steps" and event.completed >= 1:
            cancel.set()

    with gate.admit() as lease:
        session = manager.session(lease, policy(), "tests.runtime_backend:ResidentBackend")
        pins = load(session)
        identity = read_record(session.directory).worker
        with pytest.raises(WorkerStopped, match="cancelled"):
            session.run(training(session, pins, steps=30, delay=0.1), cancel=cancel, on_event=report)
        assert wait_gone(identity.pid)


def test_parent_actor_and_issued_write_survive_lost_acknowledgement(runtime, tmp_path):
    """Only the parent invokes a fake Rust hop; lost responses cannot make a write retryable."""
    manager, gate = runtime
    path = tmp_path / "effects.jsonl"
    calls = []

    def dispatch(actor, request, artifacts):
        """Simulate a committed remote write whose HTTP response is lost."""
        calls.append((actor, request.effect.operation))
        raise ConnectionError("acknowledgement lost")

    broker = EffectBroker("agents", dispatch, EffectJournal(path))
    with gate.admit() as lease:
        session = manager.session(lease, policy(), "tests.runtime_backend:Effects")
        pins = load(session)
        with pytest.raises(WorkerStopped):
            session.run(training(session, pins), effects=broker)
    assert calls == [("agents", "gym_finish")]
    identity = UUID(json.loads(path.read_text().strip())["operation_id"])
    reloaded = EffectJournal(path)
    with pytest.raises(ValueError, match="already_issued"):
        reloaded.issue(identity)
    assert len(calls) == 1


@pytest.mark.parametrize("callback_kind", ["effect", "adoption"])
def test_cancelled_worker_retains_admission_and_grants_until_parent_callback_drains(runtime, tmp_path, callback_kind):
    """Real child cancellation/reaping cannot retire artifacts or admission beneath a running effect or publication callback."""
    manager, gate = runtime
    started, release, completed = threading.Event(), threading.Event(), threading.Event()
    failures = []

    def dispatch(actor, operation, artifacts):
        """Hold the actual parent effect after issuance and materialize its response only once cleanup permits it."""
        started.set()
        assert release.wait(5)
        result = artifacts.write_bytes(b"{}", "effect_data_v1")
        completed.set()
        return result

    def adopt(notice, artifacts):
        """Keep publication input alive until the child is reaped and the parent is permitted to finish adoption."""
        started.set()
        assert release.wait(5)
        artifacts.verify(notice.artifact)
        completed.set()

    broker = EffectBroker("agents", dispatch, EffectJournal(tmp_path / "issued.jsonl"))
    with gate.admit() as lease:
        backend = "tests.runtime_backend:Effects" if callback_kind == "effect" else "tests.runtime_backend:ResidentBackend"
        session = manager.session(lease, policy(effect_seconds=10.0, artifact_adoption_seconds=10.0), backend)
        pins = load(session)
        worker = read_record(session.directory).worker
        cancel = threading.Event()

        def cancel_during_callback():
            """Observe real worker death while the parent still owns its live callback, grant directory and lease."""
            try:
                assert started.wait(3)
                cancel.set()
                assert wait_gone(worker.pid, timeout=3)
                assert session.directory.exists() and not completed.is_set()
                with pytest.raises(Busy):
                    with gate.admit():
                        pytest.fail("pending parent effect released admission")
            except BaseException as error:
                failures.append(error)
            finally:
                release.set()

        thread = threading.Thread(target=cancel_during_callback)
        thread.start()
        try:
            with pytest.raises(WorkerStopped, match="cancelled"):
                session.run(training(session, pins, checkpoint=callback_kind == "adoption"),
                    effects=broker if callback_kind == "effect" else None,
                    adopt=adopt if callback_kind == "adoption" else None, cancel=cancel)
        finally:
            release.set()
            thread.join(timeout=5)
        assert not thread.is_alive() and not failures and completed.is_set()
        assert not session.directory.exists()
    with gate.admit() as available:
        available.require_active()


def test_parent_adoption_failure_never_acknowledges_success(runtime):
    """A failed registration callback stops the complete job before another recovery file is issued."""
    manager, gate = runtime
    notices = []

    def reject(notice, artifacts):
        """Fail parent registration after the first complete staged checkpoint."""
        notices.append(notice)
        raise RuntimeError("registration failed")

    with gate.admit() as lease:
        session = manager.session(lease, policy(), "tests.runtime_backend:ResidentBackend")
        pins = load(session)
        with pytest.raises(RuntimeError, match="registration failed"):
            session.run(training(session, pins, checkpoint=True), adopt=reject)
    assert len(notices) == 1


def test_parent_adoption_timeout_is_supervised_independently(runtime):
    """Even a stuck trusted parent callback cannot keep the model worker alive beyond its deadline."""
    manager, gate = runtime
    complete = threading.Event()

    def slow_adopt(notice, artifacts):
        """Represent a slow parent-side storage call without executing any custom source."""
        complete.wait(timeout=2)

    with gate.admit() as lease:
        session = manager.session(lease, policy(artifact_adoption_seconds=0.15), "tests.runtime_backend:ResidentBackend")
        pins = load(session)
        identity = read_record(session.directory).worker
        try:
            with pytest.raises(WorkerStopped):
                session.run(training(session, pins, checkpoint=True), adopt=slow_adopt)
            assert wait_gone(identity.pid)
        finally:
            complete.set()


def test_owned_helper_is_reaped_and_excess_helpers_are_refused(runtime):
    """Actual ordinary descendants are counted and cleaned up under the same owning guardian."""
    manager, gate = runtime
    for count in (2, 1):
        limits = policy(training_progress_seconds=0.2)
        limits = limits.model_copy(update={"resources": limits.resources.model_copy(update={"owned_processes": count})})
        with gate.admit() as lease:
            session = manager.session(lease, limits, "tests.runtime_backend:Descendant")
            pins = load(session)
            with pytest.raises(WorkerStopped):
                session.run(training(session, pins))
            helpers = [event.payload.completed for event in session.history if event.payload.type == "progress" and event.payload.unit == "phases"]
            for pid in helpers:
                assert wait_gone(pid)


def test_pid_reuse_never_authorizes_killing_unrelated_process(monkeypatch):
    """A mismatched birth identity fails closed before any killpg system call."""
    real = sample_process(os.getpid()).identity
    stale = real.model_copy(update={"start_ticks": real.start_ticks - 1})
    calls = []
    monkeypatch.setattr(os, "killpg", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="pid_reused"):
        terminate_group(stale)
    assert calls == []


def test_idle_resident_expiry_releases_admission_without_another_request(runtime):
    """A guardian idle expiry is drained by the parent even when nobody asks to generate again."""
    manager, gate = runtime
    session = manager.resident_session(policy(serving_idle_seconds=0.2), "tests.runtime_backend:ResidentBackend")
    load(session)
    identity = read_record(session.directory).worker
    deadline = time.monotonic() + 5
    while not session._closed and time.monotonic() < deadline:
        time.sleep(0.02)
    assert session._closed
    assert wait_gone(identity.pid)
    with gate.admit():
        pass


def test_idle_monitor_does_not_reject_resident_load_or_generation(runtime, monkeypatch):
    """A healthy resident accepts sequential requests even when its idle monitor is paused during polling."""
    manager, gate = runtime
    session = manager.resident_session(policy(), "tests.runtime_backend:ResidentBackend")
    polling, release = threading.Event(), threading.Event()
    original_select = select.select

    def pause_idle_poll(readers, writers, errors, timeout=None):
        """Hold one real monitor poll so load and generation run under the formerly failing thread ordering."""
        if (threading.current_thread() is manager._monitor_thread
                and readers == [session.control] and timeout == 0 and not polling.is_set()):
            polling.set()
            if not release.wait(5):
                raise TimeoutError("idle monitor test was not released")
        return original_select(readers, writers, errors, timeout)

    monkeypatch.setattr(supervisor.select, "select", pause_idle_poll)
    try:
        assert polling.wait(5), "resident monitor did not poll the live worker"
        pins = load(session)
        assert session.run(generation(session, pins)).outcome == "ok"
        assert session.run(generation(session, pins, tokens=2)).outcome == "ok"
        assert not session.closed
        with pytest.raises(Busy):
            with gate.admit():
                pass
    finally:
        release.set()


def test_resident_rejects_overlapping_requests_without_retiring_active_work(runtime):
    """A second real request still fails immediately while the original request completes on the same resident."""
    manager, gate = runtime
    session = manager.resident_session(policy(inference_seconds=5), "tests.runtime_backend:ResidentBackend")
    pins = load(session)
    command = generation(session, pins)
    running, release = threading.Event(), threading.Event()

    def pause_generation(event):
        """Hold the active request at a genuine child progress event until the overlap assertion finishes."""
        if event.type == "progress":
            running.set()
            if not release.wait(5):
                raise TimeoutError("active generation test was not released")

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(session.run, command, on_event=pause_generation)
        try:
            assert running.wait(5), "generation did not report progress"
            with pytest.raises(WorkerStopped, match="one request may execute per worker"):
                session.run(command)
            assert not session.closed
        finally:
            release.set()
        assert pending.result(timeout=5).outcome == "ok"
    assert session.run(command).outcome == "ok"


def test_progress_flood_has_bounded_parent_retention(runtime):
    """A real fast producer either completes under backpressure or stops with bounded history."""
    manager, gate = runtime
    limits = policy(training_progress_seconds=2)
    with gate.admit() as lease:
        session = manager.session(lease, limits, "tests.runtime_backend:Flood")
        pins = load(session)
        identity = read_record(session.directory).worker
        try:
            session.run(training(session, pins, steps=1))
        except WorkerStopped:
            pass
        finally:
            session.close()
        assert len(session.history) <= limits.retained_progress
        assert wait_gone(identity.pid)


def test_parent_sigkill_reaps_owned_tree_before_restart(tmp_path):
    """Actual parent death closes liveness; a replacement manager verifies cleanup before admission."""
    program = """
import json,sys,threading,time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from llm.model.device import DeviceGate
from llm.runtime.supervisor import WorkerManager
from llm.runtime.processes import read_record
from tests.test_runtime_supervision import policy,load,training
gate=DeviceGate()
manager=WorkerManager(Path(sys.argv[2]),gate)
manager.start()
scope=gate.admit()
lease=scope.__enter__()
session=manager.session(lease,policy(training_progress_seconds=30),'tests.runtime_backend:Descendant')
pins=load(session)
def report(event):
    if event.type=='progress' and event.unit=='phases':
        record=read_record(session.directory)
        print(json.dumps([session.guardian.pid,record.worker.pid,event.completed]),flush=True)
threading.Thread(target=lambda:session.run(training(session,pins),on_event=report),daemon=True).start()
time.sleep(300)
"""
    root = tmp_path / "runtime"
    parent = subprocess.Popen(
        [sys.executable, "-I", "-c", program, str(Path(__file__).resolve().parents[1]), str(root)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, close_fds=True,
    )
    try:
        assert select.select([parent.stdout], [], [], 30)[0], "parent did not become ready"
        line = parent.stdout.readline()
        assert line, "parent exited before reporting owned identities"
        guardian, worker, helper = json.loads(line)
        parent.kill()
        parent.wait(timeout=2)
        assert wait_gone(worker)
        assert wait_gone(helper)
        with WorkerManager(root, DeviceGate()) as manager:
            assert not tuple(manager._sessions)
        assert not list(root.glob("*/ownership.json"))
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.communicate(timeout=5)


def test_guardian_death_persists_quarantine_across_restart(tmp_path):
    """A killed guardian cannot release admission without its authenticated cleanup certificate."""
    root = tmp_path / "runtime"
    gate = DeviceGate()
    manager = WorkerManager(root, gate)
    manager.start()
    try:
        with gate.admit() as lease:
            session = manager.session(lease, policy(), "tests.runtime_backend:ResidentBackend")
            identity = read_record(session.directory).worker
            session.guardian.kill()
            session.guardian.wait(timeout=2)
            with pytest.raises(CleanupUncertain):
                session.close()
            deadline = time.monotonic() + 2
            while identity_is_running(identity.pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            assert not identity_is_running(identity.pid)
        with pytest.raises(Busy, match="quarantined"):
            with gate.admit():
                pass
    finally:
        manager.close()
    with pytest.raises(CleanupUncertain, match="quarantined"):
        WorkerManager(root, DeviceGate()).start()


def identity_is_running(pid: int) -> bool:
    """Distinguish actual execution from an init-owned zombie after external guardian death."""
    try:
        return sample_process(pid).state != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.skipif(os.environ.get("LLM_TEST_CUDA") != "1", reason="GPU hardware verification explicitly deferred by user; set LLM_TEST_CUDA=1")
def test_cuda_worker_allocator_and_forced_cleanup(tmp_path):
    """Run the same actual resident/autograd shutdown contract on a supported CUDA device."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")
    limits = policy()
    resources = limits.resources.model_copy(update={"device": "cuda", "device_allocator_bytes": 1 << 30})
    limits = limits.model_copy(update={"resources": resources})
    gate = DeviceGate()
    with WorkerManager(tmp_path / "cuda-runtime", gate) as manager, gate.admit() as lease:
        session = manager.session(lease, limits, "tests.runtime_backend:BackwardHang")
        pins = load(session)
        identity = read_record(session.directory).worker
        with pytest.raises(WorkerStopped):
            session.run(training(session, pins))
        assert wait_gone(identity.pid)
