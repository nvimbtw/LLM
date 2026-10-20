"""Real production worker lifecycle, admission, resource and crash/restart evidence."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import select
import signal
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

    def dispatch(actor, operation, artifacts):
        """Simulate a committed remote write whose HTTP response is lost."""
        calls.append((actor, operation.operation))
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
