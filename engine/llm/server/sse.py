"""Server-sent events framing.

SSE rather than a WebSocket because the traffic is one-directional — the engine
reports, the caller listens — and cancellation is a separate POST rather than a
message on the same socket. That keeps the consumer a plain HTTP client with no
handshake to get wrong.
"""

import json
from typing import Any, AsyncIterator

from llm.server.runs import Event, Registry, Run


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
