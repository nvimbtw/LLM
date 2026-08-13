"""Server-sent events framing.

This is the listen-only path: the engine reports, the caller listens, and
cancellation is a separate POST. That keeps the consumer a plain HTTP client
with no handshake to get wrong, and it replays a run from the start, so it is
also how anything reattaches to a run already in flight.

A caller that has to talk back — the queue, which cancels — takes the socket in
`ws.py` instead and gets the same events over it.
"""

import json
from typing import Any, AsyncIterator

from api.runs import Event, Registry, Run


def frame(event: Event) -> str:
    """One wire frame. The blank line is the delimiter and is not optional."""
    return f"event: {event.kind}\ndata: {json.dumps(event.data)}\n\n"


async def run_events(registry: Registry, run: Run) -> AsyncIterator[str]:
    async for event in registry.subscribe(run):
        yield frame(event)


def token_frame(kind: str, data: dict[str, Any]) -> str:
    return f"event: {kind}\ndata: {json.dumps(data)}\n\n"


# Text/event-stream plus the headers that stop a proxy buffering the stream into
# uselessness — without these a reverse proxy can hold every frame until the
# response ends, which for a training run is the entire point missed.
HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
MEDIA_TYPE = "text/event-stream"


async def once(kind: str, payload: dict[str, Any]) -> AsyncIterator[str]:
    """A single frame and then end of stream, for a run with no live state —
    queued, or finished before anyone attached."""
    yield token_frame(kind, payload)
