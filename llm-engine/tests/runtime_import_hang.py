"""An intentionally nonreturning source import used only in fresh worker interpreters."""

import time


def hang() -> None:
    """Represent an import side effect that ignores cooperative cancellation."""
    while True:
        time.sleep(0.05)


hang()
