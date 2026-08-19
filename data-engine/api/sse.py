"""Server-sent events framing.

This is the listen-only path: the service reports, the caller listens, and
cancellation is a separate POST. It replays a run from the start, so it is also
how anything reattaches to a run already in flight.

The backend parses these frames by splitting on the blank line and reading the
`event:` and `data:` prefixes, so the shape below is a contract with it.
"""

import json
from typing import Any, AsyncIterator

from api.runs import Registry, Run


def frame(kind: str, data: dict[str, Any]) -> str:
    """One wire frame. The blank line is the delimiter and is not optional."""
    return f"event: {kind}\ndata: {json.dumps(data)}\n\n"


async def run_events(registry: Registry, run: Run) -> AsyncIterator[str]:
    async for event in registry.subscribe(run):
        yield frame(event.kind, event.data)


async def once(kind: str, payload: dict[str, Any]) -> AsyncIterator[str]:
    """A single frame and then end of stream, for a run with no live state —
    queued, or finished before anyone attached."""
    yield frame(kind, payload)


# Text/event-stream plus the headers that stop a proxy buffering the stream into
# uselessness — without these a reverse proxy can hold every frame until the
# response ends, which for a long pull is the entire point missed.
HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
MEDIA_TYPE = "text/event-stream"
