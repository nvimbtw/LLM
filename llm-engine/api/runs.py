"""What a run is doing right now.

The durable half of a run is its row; this is the half that does not survive a
restart and should not: the cancel flag a step boundary reads, the frames a
subscriber is watching, the sockets watching them. `Runner` owns the queue and
calls in here to open and close each run's live state.

Cancellation is cooperative and lands at a step boundary, where the weights are
consistent.
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from llm.architecture.wire import Diagnostic, ValidationEvent

#: Frames after which a subscriber stops listening.
FINAL = ("done", "error", "cancelled")


@dataclass
class Event:
    """One thing worth telling a subscriber about."""

    kind: str  # "train" | "progress" | "log" | "done" | "error" | "cancelled"
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

    @property
    def is_terminal(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")


class Registry:
    """The live state of every run this process has executed."""

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

    def active(self) -> Run | None:
        return next((r for r in self._runs.values() if not r.is_terminal), None)

    def adopt(self, run_id: str, kind: str, config: dict[str, Any]) -> Run:
        """Open the live state for a run the queue has just claimed."""
        run = Run(id=run_id, kind=kind, config=config)
        with self._lock:
            self._runs[run.id] = run
        return run

    def finish(self, run: Run, status: str, **fields: Any) -> None:
        """Close out a live run and tell its subscribers, once."""
        run.status = status
        run.result = fields.get("result")
        run.error = fields.get("error")
        run.finished_at = time.time()

        kind = {"completed": "done", "cancelled": "cancelled", "failed": "error"}[status]
        payload = {"error": run.error} if status == "failed" else {"result": run.result}
        if run.kind == "model_validation":
            payload = ValidationEvent(schema_version=1,run_id=UUID(run.id),sequence=len(run.history),type=kind,
                verdict=run.result.get("verdict") if status == "completed" and run.result else None,
                diagnostic=Diagnostic(code="validation_execution",phase="execution",message=(run.error or "validation failed")[:2000])
                if status == "failed" else None).model_dump(mode="json",exclude_none=True)
        self.publish(run, Event(kind, payload))

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
        for queue in queues:
            self._loop.call_soon_threadsafe(queue.put_nowait, event)

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
                if event.kind in FINAL:
                    return

            while True:
                event = await queue.get()
                yield event
                if event.kind in FINAL:
                    return
        finally:
            with run._lock:
                run.subscribers.discard(queue)


# One registry per process, matching the one device it manages.
REGISTRY = Registry()
