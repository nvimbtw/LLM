"""HTTP ownership for finite interactive work executed outside the asynchronous event loop."""

import asyncio
from collections.abc import Callable
import threading
from typing import TypeVar

import anyio
from fastapi import Request

from llm.runtime.supervisor import RequestCancelled

Result = TypeVar("Result")


async def run(request: Request, operation: Callable[[threading.Event], Result]) -> Result:
    """Run one blocking operation with owned cancellation; disconnect or task cancellation drains its worker before returning.

    The operation must observe the event and own its cleanup. Shielded draining preserves
    cleanup failures and prevents an abandoned thread from retaining numerical ownership.
    """
    cancel = threading.Event()
    pending = asyncio.create_task(asyncio.to_thread(operation, cancel))
    try:
        while not pending.done():
            await asyncio.wait({pending}, timeout=0.1)
            if not pending.done() and await request.is_disconnected():
                raise RequestCancelled("interactive client disconnected")
        return pending.result()
    finally:
        cancel.set()
        if not pending.done():
            with anyio.CancelScope(shield=True):
                await asyncio.shield(pending)
