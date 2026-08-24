import asyncio
import json
MEDIA_TYPE = "text/event-stream"
HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
def _frame(event: str, data) -> bytes:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n".encode()
async def run_events(registry, live):
    for ev, data in list(live.history):
        yield _frame(ev, data)
    queue: asyncio.Queue = asyncio.Queue()
    live.subscribe(queue)
    try:
        while True:
            ev, data = await queue.get()
            yield _frame(ev, data)
            if ev in ("done","error","cancelled"):
                break
    finally:
        live.unsubscribe(queue)
def once(event: str, data) -> bytes:
    return _frame(event, data)
