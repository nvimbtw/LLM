"""HTTP waiter ownership and cancellation drain independently of numerical process admission."""

import asyncio
import threading

import pytest

from api.interactive import run
from llm.runtime.supervisor import CleanupUncertain, RequestCancelled


class RequestState:
    """Expose a controlled disconnected state without claiming real Rust/browser transport evidence."""

    def __init__(self, *, disconnected=False):
        """Keep only the HTTP boundary's explicit observation for one request."""
        self.disconnected = disconnected

    async def is_disconnected(self):
        """Return the controlled boundary observation without side effects."""
        return self.disconnected


async def test_interactive_completion_runs_off_loop_and_returns_the_owned_result():
    """Normal responses settle their one blocking operation and keep the event loop responsive."""
    loop_thread = threading.get_ident()

    def operation(cancel):
        """Observe the owning thread and return one bounded successful result."""
        assert threading.get_ident() != loop_thread and not cancel.is_set()
        return "complete"

    assert await run(RequestState(), operation) == "complete"


async def test_disconnect_signals_and_drains_before_releasing_request():
    """An observed disconnect cannot return while its cooperative operation still owns numerical work."""
    finished = threading.Event()

    def operation(cancel):
        """Retain explicit ownership until the HTTP event arrives, then complete finite cleanup."""
        assert cancel.wait(3)
        finished.set()

    with pytest.raises(RequestCancelled):
        await run(RequestState(disconnected=True), operation)
    assert finished.is_set()


async def test_cancelled_waiter_and_failed_cleanup_preserve_owned_drain():
    """Task cancellation signals the operation and a cleanup uncertainty is never hidden by the cancelled waiter."""
    started, finished = threading.Event(), threading.Event()

    def operation(cancel):
        """Report explicit cleanup uncertainty after the request owner signals cancellation."""
        started.set()
        assert cancel.wait(3)
        finished.set()
        raise CleanupUncertain("test cleanup uncertainty")

    pending = asyncio.create_task(run(RequestState(), operation))
    while not started.is_set():
        await asyncio.sleep(.01)
    pending.cancel()
    with pytest.raises(CleanupUncertain, match="uncertainty"):
        await pending
    assert finished.is_set()
