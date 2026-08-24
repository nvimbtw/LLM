import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import store
from api.runs import REGISTRY
from store import queries as q
from store.schema import RunKind, RunStatus
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bench-runner")
class Progress:
    def __init__(self, run_id: uuid.UUID):
        self.run_id = run_id
        self._last = 0.0
    def __call__(self, *, stage: str | None = None, progress: float | None = None, detail: str | None = None):
        now = time.monotonic()
        if now - self._last < 2.0 and progress not in (1.0, None):
            return
        self._last = now
        with store.session() as db:
            run = db.get(q.Run, self.run_id)
            if run is None:
                return
            if stage is not None:
                run.stage = stage
            if progress is not None:
                run.progress = max(0.0, min(1.0, progress))
            db.flush()
        REGISTRY.publish(str(self.run_id), "progress", {"stage": stage, "progress": progress, "fraction": progress, "detail": detail})
class Runner:
    def __init__(self, registry, handlers: dict):
        self.registry = registry
        self.handlers = handlers
        self._wake = asyncio.Event()
    def enqueue(self, kind: RunKind, *, label: str, config: dict, owner_id=None, **fields) -> uuid.UUID:
        with store.session() as db:
            run = q.enqueue(db, kind=kind, label=label, config=config, owner_id=owner_id, **fields)
            rid = run.id
        self.wake()
        return rid
    def cancel(self, run_id: uuid.UUID) -> bool:
        with store.session() as db:
            run = db.get(q.Run, run_id)
            if run is None or run.status.is_terminal:
                return False
            if run.status == q.RunStatus.queued or run.status == q.RunStatus.waiting:
                run.status = q.RunStatus.cancelled
                run.finished_at = q._now()
                return True
            if run.status == q.RunStatus.running:
                run.status = q.RunStatus.cancelling
                return True
            return False
    def wake(self):
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(self._wake.set)
        except RuntimeError:
            self._wake.set()
    async def wait(self, run_id: uuid.UUID):
        while True:
            with store.session() as db:
                run = db.get(q.Run, run_id)
                if run is None or run.status.is_terminal:
                    return
            await asyncio.sleep(0.5)
    async def drain_forever(self):
        while True:
            await self._drain_once()
            await self._sleep()
    async def _drain_once(self):
        with store.session() as db:
            run = q.claim_next(db)
            if run is None:
                return
            run_id = run.id
            kind = run.kind
            config = dict(run.config)
            label = run.label
        live = self.registry.adopt(str(run_id))
        live.emit("start", {"run_id": str(run_id), "kind": kind.value, "label": label})
        handler = self.handlers.get(kind)
        if handler is None:
            with store.session() as db:
                q.finish_run(db, run_id, RunStatus.failed, error=f"no handler for {kind}")
            live.emit("error", {"error": f"no handler for {kind}"})
            return
        cancel = threading.Event()
        def check_cancel():
            with store.session() as db:
                r = db.get(q.Run, run_id)
                return r is not None and r.status == RunStatus.cancelling
        progress = Progress(run_id)
        try:
            result = await asyncio.to_thread(handler, config, progress, cancel, None)
            is_cancelled = check_cancel()
            if is_cancelled or cancel.is_set():
                with store.session() as db:
                    q.finish_run(db, run_id, RunStatus.cancelled)
                live.emit("cancelled", {"run_id": str(run_id)})
                return
            with store.session() as db:
                q.finish_run(db, run_id, RunStatus.done, result=result, ref_id=result.get("benchmark_id") if isinstance(result, dict) else None)
                for dep in q.waiting_on(db, run_id):
                    dep.status = RunStatus.queued
            live.emit("done", {"run_id": str(run_id), "result": result})
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            with store.session() as db:
                q.finish_run(db, run_id, RunStatus.failed, error=f"{e}\n{tb}")
            live.emit("error", {"error": str(e)})
    async def _sleep(self):
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        self._wake.clear()
