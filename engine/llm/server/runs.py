"""The run registry: one device, one line.

A pipeline is long-running, so a route cannot simply call it and return — the
caller needs progress while it works and a way to stop it. `start` puts the
pipeline on a worker thread and hands back a `Run`; progress frames are
published to every subscriber and also kept, so a consumer that connects late
(or reconnects) sees the whole run rather than the tail.

Cancellation is cooperative and lands at a step boundary, where the weights are
consistent — the same contract the Rust queue had.

There is one GPU, so there is one runner and one line: `start` refuses while
another run is live rather than letting two runs fight over the device and both
finish slower than either would have alone.
"""

import asyncio
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

# Statuses a run never leaves.
TERMINAL = frozenset({"completed", "failed", "cancelled"})


class EngineBusy(RuntimeError):
    """Raised when a run is asked for while the device is already occupied."""


@dataclass
class Event:
    """One thing worth telling a subscriber about."""

    kind: str  # "progress" | "log" | "done" | "error" | "cancelled"
    data: dict[str, Any]


@dataclass
class Run:
    id: str
    kind: str
    config: dict[str, Any]
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    # Set from the API thread, read by the training loop at each step boundary.
    cancel: threading.Event = field(default_factory=threading.Event)

    # Guards `history` and `subscribers` together: a subscriber must be able to
    # attach and copy the backlog without a frame slipping between the two and
    # being lost.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    history: list[Event] = field(default_factory=list, repr=False)
    subscribers: set[asyncio.Queue] = field(default_factory=set, repr=False)

    # Set on the event loop when the run reaches a terminal state, so `wait`
    # does not have to poll.
    _finished: asyncio.Event | None = field(default=None, repr=False)

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_secs": round((self.finished_at or time.time()) - self.started_at, 1),
            "result": self.result,
            "error": self.error,
            "frames": len(self.history),
        }

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL


class Registry:
    """Every run this process has executed, and the one that is live."""

    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the event loop, so worker threads can wake subscribers on it.

        Called once at app startup. Publishing from a thread without this would
        touch asyncio primitives from the wrong thread, which is exactly the
        class of bug that shows up as a hang rather than an exception.
        """
        self._loop = loop

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def list(self) -> list[dict[str, Any]]:
        return [r.snapshot() for r in sorted(self._runs.values(), key=lambda r: r.started_at)]

    def active(self) -> Run | None:
        return next((r for r in self._runs.values() if not r.is_terminal), None)

    def start(
        self,
        kind: str,
        config: dict[str, Any],
        target: Callable[[Callable[[Any], None], threading.Event], dict[str, Any]],
    ) -> Run:
        """Run `target` on a worker thread and return its handle immediately.

        `target` is called with an `on_progress` callback and a cancel event —
        the same two seams the pipelines already take.
        """
        with self._lock:
            if (busy := self.active()) is not None:
                raise EngineBusy(
                    f"run {busy.id} ({busy.kind}) is still active; there is one device"
                )
            run = Run(id=uuid.uuid4().hex[:12], kind=kind, config=config)
            if self._loop is not None:
                run._finished = asyncio.Event()
            self._runs[run.id] = run

        threading.Thread(
            target=self._execute, args=(run, target), name=f"run-{run.id}", daemon=True
        ).start()
        return run

    def _execute(self, run: Run, target) -> None:
        """The worker thread body. Never raises — a run's failure is data."""
        try:
            result = target(lambda p: self.publish(run, Event("progress", _as_dict(p))), run.cancel)

            if run.cancel.is_set() or (isinstance(result, dict) and result.get("cancelled")):
                run.status = "cancelled"
                run.result = result
                final = Event("cancelled", {"result": result})
            else:
                run.status = "completed"
                run.result = result
                final = Event("done", {"result": result})
        except Exception as e:
            run.status = "failed"
            run.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            final = Event("error", {"error": run.error})

        run.finished_at = time.time()
        self.publish(run, final)

        if run._finished is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(run._finished.set)

    def publish(self, run: Run, event: Event) -> None:
        """Record an event and hand it to every subscriber.

        Called from the worker thread, so the queues are fed through
        `call_soon_threadsafe` rather than touched directly.
        """
        with run._lock:
            run.history.append(event)
            queues = list(run.subscribers)

        if self._loop is None:
            return
        for q in queues:
            self._loop.call_soon_threadsafe(q.put_nowait, event)

    async def subscribe(self, run: Run):
        """Yield every event of a run: the backlog first, then live ones, then
        stop once it is terminal.

        The backlog is copied under the same lock that registers the queue, so a
        frame emitted mid-subscribe lands in exactly one of the two and is
        neither lost nor duplicated.
        """
        queue: asyncio.Queue = asyncio.Queue()
        with run._lock:
            backlog = list(run.history)
            run.subscribers.add(queue)

        try:
            for event in backlog:
                yield event
                if event.kind in ("done", "error", "cancelled"):
                    return

            while True:
                event = await queue.get()
                yield event
                if event.kind in ("done", "error", "cancelled"):
                    return
        finally:
            with run._lock:
                run.subscribers.discard(queue)

    async def wait(self, run: Run) -> Run:
        """Block until a run reaches a terminal state."""
        if run._finished is not None:
            await run._finished.wait()
        return run


def _as_dict(progress: Any) -> dict[str, Any]:
    """Progress objects are dataclasses; anything else is passed through."""
    if hasattr(progress, "__dict__"):
        data = dict(vars(progress))
        if hasattr(progress, "fraction"):
            data["fraction"] = round(progress.fraction, 4)
        return data
    return {"value": progress}


# One registry per process, matching the one device it manages.
REGISTRY = Registry()
