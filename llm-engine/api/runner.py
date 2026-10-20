"""The queue: durable rows, one runner, one device.

`runs.Registry` owns what a run is doing *right now* — the cancel flag, the
frame history, the sockets watching it. None of that survives a restart and
none of it should: a frame is only interesting while someone is attached, and
writing loss curves per step would flatten the database for nothing.

What survives is the queue. A run is a row before it starts and after it ends,
so closing the tab does not lose the reply and a restart resumes the line
rather than dropping it.
"""

import asyncio
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import store
from api import chain, handlers
from api.handlers import HANDLERS
from api.runs import REGISTRY, Event, Registry, Run
from api.worker import SupervisedHandler, SupervisedRun
from llm.orchestrate import chat as chat_pipeline
from llm.model import device
from llm.train.loop import release_device
from llm.train.progress import Checkpoint, Progress, Tokenize
from llm.runtime.supervisor import RequestCancelled, WorkerManager
from store import queries as q
from store.schema import RunKind, RunStatus

#: Given a config, report progress, check for cancellation and say who the run
#: is for, return a result. The first two are the seams the pipelines already
#: take; the owner is there because phase 1 reads its documents through the
#: backend, which scopes them to a person.
Handler = Callable[[dict, Callable[[Any], None], threading.Event, str | None], dict]

#: How often the drain loop looks for work when nothing woke it. Everything
#: that enqueues also wakes it, so this only covers a missed notification.
IDLE_POLL_SECS = 2.0

#: How often a running job's coarse progress reaches its row. The live frames
#: go out at whatever rate the worker produces them; this is only so a fresh
#: page load has a sane bar before its socket attaches.
PROGRESS_WRITE_SECS = 2.0


class Runner:
    """Drains the queue, one run at a time."""

    def __init__(self, registry: Registry, handlers: dict[RunKind, Handler | SupervisedHandler]) -> None:
        """Own one durable queue consumer, retaining optional supervised runtime dependencies explicitly."""
        self._registry = registry
        self._handlers = handlers
        self._wake: asyncio.Event | None = None
        self._current: uuid.UUID | None = None
        self._paused = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._workers: WorkerManager | None = None
        self._worker_future: asyncio.Task | None = None

    def attach_workers(self, workers: WorkerManager) -> None:
        """Install the started parent runtime owner before queue draining begins."""
        if self._current is not None:
            raise RuntimeError("cannot replace runtime ownership while a job is active")
        self._workers = workers

    async def shutdown(self) -> None:
        """Stop supervised processes first, then drain the actual active thread before loop cancellation.

        Legacy builtin threads remain cooperative; their cancellation event is set
        and a bounded wait never pretends to terminate their executing code.
        """
        self._paused = True
        live = self._registry.get(str(self._current)) if self._current else None
        if live is not None:
            live.cancel.set()
        if self._workers is not None:
            await asyncio.to_thread(self._workers.close)
        future = self._worker_future
        if future is not None:
            try:
                await asyncio.wait_for(asyncio.shield(future), timeout=10)
            except TimeoutError:
                print("llm-engine: cooperative legacy worker is still stopping")
            except Exception:
                pass

    # ── queueing ────────────────────────────────────────────────────────────

    def enqueue(
        self,
        kind: RunKind,
        *,
        label: str,
        config: dict,
        owner_id: uuid.UUID | None = None,
        **fields,
    ) -> uuid.UUID:
        with store.session() as db:
            run = q.enqueue(
                db, kind=kind, label=label, config=config, owner_id=owner_id, **fields
            )
            run_id = run.id
        self.wake()
        return run_id

    def wake(self) -> None:
        """Ask the drain loop to look now rather than at the next idle poll.

        Silent when nothing is draining — the loop rereads the queue when it
        starts, so there is nothing to miss.
        """
        if self._wake is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._wake.set)

    @property
    def paused(self) -> bool:
        return self._paused

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        if not paused:
            self.wake()

    def cancel(self, run_id: uuid.UUID) -> bool:
        """Ask a run to stop. Queued rows stop immediately; a running one stops
        at its next step boundary, which is a real interval and a real state."""
        with store.session() as db:
            run = db.get(store.Run, run_id)
            if run is None or run.status.is_terminal:
                return False
            if run.status in (RunStatus.queued, RunStatus.waiting):
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
        # rather than in __init__ — an asyncio.Event binds to the first loop
        # that waits on it and refuses every one after.
        self._loop = asyncio.get_running_loop()
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
            self._loop = None

    async def _drain_once(self) -> None:
        while (claimed := self._claim()) is not None:
            run_id, kind, config, owner = claimed
            await self._execute(run_id, kind, config, owner)

    def _claim(self) -> tuple[uuid.UUID, RunKind, dict, str | None] | None:
        if self._paused:
            return None
        with store.session() as db:
            run = q.claim_next(db)
            if run is None:
                return None
            owner = str(run.owner_id) if run.owner_id else None
            return run.id, run.kind, dict(run.config or {}), owner

    async def _execute(
        self, run_id: uuid.UUID, kind: RunKind, config: dict, owner: str | None = None
    ) -> None:
        """Execute one admitted job and persist terminal state only after its worker lifetime drains.

        Failed thread tracebacks are cleared because a retained future otherwise
        keeps parent legacy tensors alive during allocator release.
        """
        config = {**config, "execution_id": str(run_id)} if kind == RunKind.benchmark else config
        handler = self._handlers.get(kind)
        if handler is None:
            self._finish(run_id, RunStatus.failed, error=f"no handler for {kind.value}")
            return

        self._current = run_id
        live = self._registry.adopt(str(run_id), kind.value, config)
        progress = _ThrottledProgress(self._registry, live, run_id)

        try:
            self._worker_future = asyncio.create_task(asyncio.to_thread(
                _work, handler, config, progress, live.cancel, owner, self._workers, run_id,
            ))
            result = await asyncio.shield(self._worker_future)
            produced = _produced(result)
            if kind in (RunKind.train, RunKind.rl) and produced is not None:
                handlers.attach_run(run_id, produced)
            if live.cancel.is_set():
                self._finish(run_id, RunStatus.cancelled, result=result, ref_id=produced)
                self._registry.finish(live, "cancelled", result=result)
            else:
                self._finish(
                    run_id,
                    RunStatus.done,
                    result=result,
                    ref_id=produced,
                    progress=1.0,
                )
                self._registry.finish(live, "completed", result=result)
        except RequestCancelled:
            self._finish(run_id, RunStatus.cancelled)
            self._registry.finish(live, "cancelled")
        except Exception as e:
            traceback.clear_frames(e.__traceback__)
            self._registry.finish(live, "failed", error=f"{type(e).__name__}: {e}")
            self._finish(run_id, RunStatus.failed, error=f"{type(e).__name__}: {e}")
        finally:
            self._current = None
            if self._worker_future is not None and self._worker_future.done():
                self._worker_future = None

    def _finish(self, run_id: uuid.UUID, status: RunStatus, **fields) -> None:
        with store.session() as db:
            finished = q.finish_run(db, run_id, status, **fields)
            produced = fields.get("ref_id") if status == RunStatus.done else None
            if finished is not None and finished.kind == RunKind.benchmark and status == RunStatus.done:
                produced = uuid.UUID(finished.config["model_name"])
            elif finished is not None and finished.kind == RunKind.tokenize:
                produced = None
        if chain.promote(run_id, produced):
            self.wake()

    async def wait(self, run_id: uuid.UUID, poll_secs: float = 0.05) -> None:
        """Block until a run reaches a terminal state.

        Polls the row rather than waiting on the live handle: a run that is
        still queued has no handle yet, and that is exactly when a caller most
        wants to wait.
        """
        while True:
            with store.session() as db:
                run = db.get(store.Run, run_id)
                if run is None or run.status.is_terminal:
                    return
            await asyncio.sleep(poll_secs)

    # ── introspection ───────────────────────────────────────────────────────

    @property
    def current(self) -> uuid.UUID | None:
        return self._current


def _work(handler, config, progress, cancel, owner, workers=None, run_id=None):
    """Preserve builtin handler behavior while handing supervised work the actual parent lease.

    Benchmark dispatch holds no device. Queue priority evicts an idle resident
    worker before admission; legacy cache cleanup stays in this parent thread.
    """
    if handler is handlers.benchmark:
        return handler(config, progress, cancel, owner)
    if workers is not None:
        workers.evict_for_queue()
    with device.admit(wait=True) as lease:
        if cancel.is_set():
            return {"cancelled": True}
        chat_pipeline.evict()
        try:
            if isinstance(handler, SupervisedHandler):
                if workers is None or run_id is None:
                    raise RuntimeError("supervised handler has no parent runtime owner")
                return handler.run_supervised(SupervisedRun(
                    job_id=run_id, config=config, actor=owner, progress=progress,
                    cancel=cancel, lease=lease, workers=workers,
                ))
            return handler(config, progress, cancel, owner)
        except Exception as error:
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            release_device()


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

        if isinstance(payload, Checkpoint):
            # Written now, not on the timer below. That throttle exists so a
            # progress bar does not cost a row per step; a checkpoint frame it
            # swallowed would be weights on disk that nothing can name.
            if handlers.record_checkpoint(self._run_id, payload) is None and not payload.final:
                raise RuntimeError("Recovery point was saved but registration failed; stopping to preserve bounded recovery storage")
            return

        now = time.monotonic()
        if now - self._last_write < PROGRESS_WRITE_SECS:
            return
        self._last_write = now
        with store.session() as db:
            run = db.get(store.Run, self._run_id)
            if run is None:
                return
            # Phase 1 has its own `progress`, deliberately not written here: the
            # row's bar is how far the *run* is, and filling it while the dataset
            # builds would send it back to zero the moment training began.
            if isinstance(frame.get("fraction"), (int, float)):
                run.progress = float(frame["fraction"])
            label = _stage_label(kind, frame)
            if label is not None:
                run.stage = label


def _stage_label(kind: str, frame: dict) -> str | None:
    """What the board calls the phase this frame came from.

    A training frame names its curriculum stage in `stage_name` — `stage` there
    is the index the charts group by, and writing that would put a number in a
    column rendered as a label. A run with no curriculum names no stage, and
    "training" is what it is doing.
    """
    if kind == "train":
        name = frame.get("stage_name")
        return name if isinstance(name, str) else "training"
    if kind == "tokenize":
        return "tokenizing"
    if kind == "rl":
        return "practising"
    stage = frame.get("stage")
    return stage if isinstance(stage, str) else None


def _frame(payload: Any) -> tuple[str, dict]:
    """One report as its consumer reads it.

    A training report arrives as the `Progress` dataclass, and goes out under its
    own name: its fields are the frontend's `TrainStats` verbatim, so it travels
    from the loop to the chart untranslated. Folding it into the generic
    `progress` frame is what left a run with no statistics and no diagnostics —
    the payload became the dataclass's `repr` and every field arrived undefined.
    A `Tokenize` is the same arrangement for the phase that builds the data, and
    a `Checkpoint` is the announcement of a model the run has already written to
    disk — the one frame whose consumer is a database row rather than a chart.

    Everything else already reports a plain dict — a stage line — and keeps
    `progress`.
    """
    if isinstance(payload, Progress):
        return "train", {**asdict(payload), "fraction": payload.fraction}
    if isinstance(payload, Tokenize):
        # A field this phase does not measure is absent, not zero — see `Tokenize`.
        return "tokenize", {k: v for k, v in asdict(payload).items() if v is not None}
    if isinstance(payload, Checkpoint):
        return "checkpoint", asdict(payload)
    if isinstance(payload, dict):
        return "progress", payload
    return "log", {"line": str(payload)}


#: One device, so one runner. Routes enqueue onto this; the lifespan drains it.
RUNNER = Runner(REGISTRY, HANDLERS)


def _produced(result: dict | None) -> uuid.UUID | None:
    """What a finished run made, when it made something with its own row."""
    if not isinstance(result, dict):
        return None
    # Ordered, not a set: a *training* result also carries the `tokenizer_id` it
    # consumed, and `ref_id` is what a run produced. Reading that one first would
    # point every training row at a vocabulary.
    for key in ("model_id", "benchmark_id", "tokenizer_id"):
        value = result.get(key)
        if value:
            try:
                return uuid.UUID(str(value))
            except ValueError:
                return None
    return None
