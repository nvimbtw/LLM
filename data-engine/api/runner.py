"""The queue: durable rows, one runner.

`runs.Registry` owns what a run is doing *right now* — the cancel flag, the frame
history, the sockets watching it. None of that survives a restart and none of it
should. What survives is the queue: a run is a row before it starts and after it
ends, so closing the tab does not lose a pull and a restart resumes the line.

One at a time, like the engine's, but for a different reason. Nothing here
contends for a device; it is the Hub that is worth being polite to, and a
serial queue is the simplest thing that cannot accidentally open twenty parallel
downloads.
"""

import asyncio
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

import store
from api.handlers import HANDLERS
from api.runs import REGISTRY, Event, Registry, Run
from store import queries as q
from store.schema import RunKind, RunStatus

#: Given a config, report progress and check for cancellation, return a result.
Handler = Callable[[dict, Callable[[Any], None], threading.Event], dict]

#: How often the drain loop looks for work when nothing woke it. Everything that
#: enqueues also wakes it, so this only covers a missed notification.
IDLE_POLL_SECS = 2.0

#: How often a running job's coarse progress reaches its row. The live frames go
#: out at whatever rate the worker produces them; this is only so a fresh page
#: load has a sane bar before its socket attaches.
PROGRESS_WRITE_SECS = 2.0


class Runner:
    """Drains the queue, one run at a time."""

    def __init__(self, registry: Registry, handlers: dict[RunKind, Handler]) -> None:
        self._registry = registry
        self._handlers = handlers
        self._wake: asyncio.Event | None = None
        self._current: uuid.UUID | None = None

    # ── queueing ────────────────────────────────────────────────────────────

    def enqueue(
        self,
        kind: RunKind,
        *,
        label: str,
        config: dict,
        owner_id: uuid.UUID | None = None,
        ref_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        with store.session() as db:
            run = q.create_run(
                db, kind=kind, label=label, config=config, owner_id=owner_id, ref_id=ref_id
            )
            run_id = run.id
        self.wake()
        return run_id

    def wake(self) -> None:
        """Ask the drain loop to look now rather than at the next idle poll.

        Silent when nothing is draining — the loop rereads the queue when it
        starts, so there is nothing to miss.
        """
        if self._wake is not None:
            self._wake.set()

    def cancel(self, run_id: uuid.UUID) -> bool:
        """Ask a run to stop. Queued rows stop immediately; a running one stops
        at its next batch boundary, which is a real interval and a real state."""
        with store.session() as db:
            run = db.get(store.Run, run_id)
            if run is None or run.status.is_terminal:
                return False
            if run.status == RunStatus.queued:
                run.status = RunStatus.cancelled
                return True
            run.status = RunStatus.cancelling

        live = self._registry.get(str(run_id))
        if live is not None:
            live.cancel.set()
        return True

    # ── the loop ────────────────────────────────────────────────────────────

    async def drain_forever(self) -> None:
        # The event belongs to whichever loop is draining, so it is made here
        # rather than in __init__ — an asyncio.Event binds to the first loop that
        # waits on it and refuses every one after.
        self._wake = asyncio.Event()
        try:
            while True:
                try:
                    await self._drain_once()
                except Exception as e:  # a broken run must not stop the queue
                    print(f"runner: {type(e).__name__}: {e}")
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=IDLE_POLL_SECS)
                except TimeoutError:
                    pass
                self._wake.clear()
        finally:
            self._wake = None

    async def _drain_once(self) -> None:
        while (claimed := self._claim()) is not None:
            run_id, kind, config = claimed
            await self._execute(run_id, kind, config)

    def _claim(self) -> tuple[uuid.UUID, RunKind, dict] | None:
        with store.session() as db:
            run = q.claim_next(db)
            if run is None:
                return None
            return run.id, run.kind, dict(run.config or {})

    async def _execute(self, run_id: uuid.UUID, kind: RunKind, config: dict) -> None:
        handler = self._handlers.get(kind)
        if handler is None:
            self._finish(run_id, RunStatus.failed, error=f"no handler for {kind.value}")
            return

        self._current = run_id
        live = self._registry.adopt(str(run_id), kind.value, config)
        progress = _ThrottledProgress(self._registry, live, run_id)

        try:
            result = await asyncio.to_thread(handler, config, progress, live.cancel)
        except Exception as e:
            self._registry.finish(live, "failed", error=f"{type(e).__name__}: {e}")
            self._finish(run_id, RunStatus.failed, error=f"{type(e).__name__}: {e}")
        else:
            if live.cancel.is_set():
                self._registry.finish(live, "cancelled", result=result)
                self._finish(run_id, RunStatus.cancelled, result=result)
            else:
                self._registry.finish(live, "completed", result=result)
                self._finish(run_id, RunStatus.done, result=result)
        finally:
            self._current = None

    def _finish(self, run_id: uuid.UUID, status: RunStatus, **fields) -> None:
        with store.session() as db:
            q.finish_run(db, run_id, status, **fields)

    async def wait(self, run_id: uuid.UUID, poll_secs: float = 0.05) -> None:
        """Block until a run reaches a terminal state.

        Polls the row rather than waiting on the live handle: a run that is still
        queued has no handle yet, and that is exactly when a caller most wants to
        wait.
        """
        while True:
            with store.session() as db:
                run = db.get(store.Run, run_id)
                if run is None or run.status.is_terminal:
                    return
            await asyncio.sleep(poll_secs)

    @property
    def current(self) -> uuid.UUID | None:
        return self._current


class _ThrottledProgress:
    """Publishes every frame live, writes a coarse one to the row on a timer."""

    def __init__(self, registry: Registry, live: Run, run_id: uuid.UUID) -> None:
        self._registry = registry
        self._live = live
        self._run_id = run_id
        self._last_write = 0.0

    def __call__(self, payload: Any) -> None:
        kind, frame = _frame(payload)
        self._registry.publish(self._live, Event(kind, frame))

        now = time.monotonic()
        if now - self._last_write < PROGRESS_WRITE_SECS:
            return
        self._last_write = now

        fraction = frame.get("fraction")
        stage = frame.get("stage")
        with store.session() as db:
            q.report_progress(
                db,
                self._run_id,
                float(fraction) if isinstance(fraction, (int, float)) else None,
                stage if isinstance(stage, str) else None,
            )


def _frame(payload: Any) -> tuple[str, dict]:
    """One report as its consumer reads it.

    A dict is a structured progress frame and travels as itself; anything else is
    a line someone printed and travels as a log, rather than arriving at the
    frontend as a dataclass `repr` with every field undefined.
    """
    if isinstance(payload, dict):
        return "progress", payload
    return "log", {"line": str(payload)}


#: Routes enqueue onto this; the lifespan drains it.
RUNNER = Runner(REGISTRY, HANDLERS)
