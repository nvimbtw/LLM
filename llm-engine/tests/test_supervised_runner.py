"""Real queue-to-worker execution commits terminal state after the owned process has stopped."""

import asyncio
import json
from pathlib import Path
import threading

import pytest

import store
from api.runner import Runner
from api.runs import Registry
from api.worker import SupervisedRun
from llm.runtime.processes import read_record
from llm.runtime.supervisor import WorkerManager
from store.schema import RunKind, RunStatus
from tests.test_runtime_supervision import load, policy, training


class FixtureHandler:
    """Run actual resident torch training behind the production supervised-handler seam."""

    def __init__(self, *, cancel_at_step: bool = False) -> None:
        """Choose finite success or cancellation after an observed optimizer step."""
        self.cancel_at_step = cancel_at_step
        self.pid: int | None = None
        self.entered = threading.Event()

    def run_supervised(self, context: SupervisedRun) -> dict:
        """Consume the actual runner lease and return only after the session's cleanup certificate."""
        with context.workers.session(context.lease, policy(), "tests.runtime_backend:ResidentBackend") as session:
            self.pid = read_record(session.directory).worker.pid
            pins = load(session)
            self.entered.set()
            result = session.run(
                training(session, pins, steps=30 if self.cancel_at_step else 2, delay=0.02),
                request_id=context.job_id, on_event=lambda event: self.report(event, context),
                cancel=context.cancel,
            )
            return json.loads(session.artifacts.read_bytes(result.result))

    def report(self, event, context: SupervisedRun) -> None:
        """Forward real numerical progress and optionally request cancellation after a completed update."""
        if event.type == "progress":
            context.progress({"fraction": min(event.completed / 30, 0.99)})
            if self.cancel_at_step and event.unit == "steps" and event.completed >= 1:
                context.cancel.set()


@pytest.mark.parametrize("cancelled", [False, True])
async def test_queue_terminal_state_follows_real_process_cleanup(store_session, tmp_path, cancelled):
    """The durable row and terminal event appear only after the real child has been reaped."""
    registry = Registry()
    registry.bind_loop(asyncio.get_running_loop())
    handler = FixtureHandler(cancel_at_step=cancelled)
    with WorkerManager(tmp_path / "runtime") as manager:
        runner = Runner(registry, {RunKind.train: handler})
        runner.attach_workers(manager)
        run_id = runner.enqueue(RunKind.train, label="supervised", config={})
        task = asyncio.create_task(runner._drain_once())
        assert await asyncio.to_thread(handler.entered.wait, 20)
        with store.session() as db:
            assert db.get(store.Run, run_id).status == RunStatus.running
        await asyncio.wait_for(task, 20)
        assert not Path(f"/proc/{handler.pid}").exists()
        with store.session() as db:
            expected = RunStatus.cancelled if cancelled else RunStatus.done
            assert db.get(store.Run, run_id).status == expected
        terminal = registry.get(str(run_id)).history[-1]
        assert terminal.kind == ("cancelled" if cancelled else "done")
        assert not list(manager.root.glob("*/ownership.json"))


async def test_startup_reconciliation_failure_releases_runtime_owner(store_session, tmp_path, monkeypatch):
    """Failure before application yield must drain the manager and release the cross-process lock."""
    import main

    root = tmp_path / "runtime"
    monkeypatch.setattr(main.store, "configure", lambda: None)
    monkeypatch.setattr(main.storage, "runtime_root", lambda: root)

    def fail_recovery() -> None:
        """Inject a parent-owned startup storage failure before queue draining starts."""
        raise RuntimeError("reconciliation failed")

    monkeypatch.setattr(main.recovery, "reconcile", fail_recovery)
    monkeypatch.setattr(main, "RUNNER", Runner(Registry(), {}))
    with pytest.raises(RuntimeError, match="reconciliation failed"):
        async with main.lifespan(main.app):
            pytest.fail("startup must not succeed")
    with WorkerManager(root) as replacement:
        assert replacement.lock_fd >= 0
