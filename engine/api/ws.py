"""A whole run over one WebSocket.

The caller opens the socket, sends its config, and receives the run id followed
by every frame the run emits until it reaches a terminal one. Cancelling is a
message on the same socket rather than a separate HTTP call, so there is no
second connection to open, authenticate or time out mid-run.

`sse.py` still serves the same events over plain HTTP for anything that only
listens. This is the path for a caller that also has to talk back.
"""

import asyncio
from typing import Any, Callable

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from api.runs import EngineBusy, Registry, Run

# What a run is started with: the same two seams `Registry.start` takes.
Pipeline = Callable[[Any], Callable[..., dict[str, Any]]]


async def serve_run(
    websocket: WebSocket,
    registry: Registry,
    kind: str,
    config_model: type[BaseModel],
    pipeline: Pipeline,
) -> None:
    """Start a run from the socket's opening frame and stream it to its end.

    Frames out are the registry's own events with their kind flattened onto a
    `type` field, so a consumer switches on one key and reads the payload it
    already knows from the event stream.
    """
    await websocket.accept()

    try:
        opening = await websocket.receive_json()
    except (WebSocketDisconnect, ValueError):
        # Gone again, or opened with something that is not JSON. Either way
        # there is no run to start and nothing to report it to.
        return

    config = opening.get("config") if isinstance(opening, dict) else None
    try:
        cfg = config_model.model_validate(config)
    except ValidationError as e:
        await _reject(websocket, str(e))
        return

    try:
        run = registry.start(kind, cfg.model_dump(), pipeline(cfg))
    except EngineBusy as e:
        await _reject(websocket, str(e))
        return

    await websocket.send_json({"type": "started", **run.snapshot()})

    control = asyncio.create_task(_control(websocket, run))
    try:
        async for event in registry.subscribe(run):
            await websocket.send_json({"type": event.kind, **event.data})
    except WebSocketDisconnect:
        # Hanging up does not end the run — it holds the device and carries on.
        # Its id was sent in the opening frame and its whole history is still on
        # /runs/{id}/events for whoever reattaches.
        pass
    finally:
        control.cancel()
        await _close(websocket)


async def _control(websocket: WebSocket, run: Run) -> None:
    """Read the caller's side of the socket for as long as the run lasts.

    Only cancellation is spoken here. It sets the same flag the HTTP cancel
    route sets, so it lands at the same step boundary, where the weights are
    consistent.
    """
    while True:
        try:
            message = await websocket.receive_json()
        except (WebSocketDisconnect, RuntimeError):
            return
        except ValueError:
            continue  # Not JSON. Not a reason to stop listening.
        if isinstance(message, dict) and message.get("type") == "cancel":
            run.cancel.set()


async def _reject(websocket: WebSocket, detail: str) -> None:
    """Refuse before a run exists, in the frame shape a failure already uses."""
    await websocket.send_json({"type": "error", "error": detail})
    await _close(websocket)


async def _close(websocket: WebSocket) -> None:
    try:
        await websocket.close()
    except RuntimeError:
        pass  # Already closed from the other end.
