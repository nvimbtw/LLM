import asyncio
from dataclasses import dataclass, field
import time
@dataclass
class LiveRun:
    id: str
    history: list[tuple[str, dict]] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    def emit(self, event: str, data: dict) -> None:
        self.history.append((event, data))
        for q in list(self.subscribers):
            try:
                q.put_nowait((event, data))
            except asyncio.QueueFull:
                pass
    def subscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.append(q)
    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self.subscribers:
            self.subscribers.remove(q)
class Registry:
    def __init__(self):
        self._runs: dict[str, LiveRun] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
    def get(self, run_id: str) -> LiveRun | None:
        return self._runs.get(run_id)
    def adopt(self, run_id: str) -> LiveRun:
        live = LiveRun(id=run_id)
        self._runs[run_id] = live
        return live
    def active(self) -> LiveRun | None:
        for r in self._runs.values():
            if r.history and r.history[-1][0] not in ("done","error","cancelled"):
                return r
        return None
    def publish(self, run_id: str, event: str, data: dict) -> None:
        live = self._runs.get(run_id)
        if live is None:
            return
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(live.emit, event, data)
        else:
            live.emit(event, data)
REGISTRY = Registry()
