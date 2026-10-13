"""One admission gate for execution and cache eviction in the LLM process."""
from contextlib import contextmanager
from threading import Condition

_condition = Condition()
_busy = False
_waiters = 0


class Busy(ValueError):
    pass


@contextmanager
def admit(*, wait: bool = False):
    global _busy, _waiters
    with _condition:
        if not wait and (_busy or _waiters):
            raise Busy('The device is busy; retry after the active execution finishes')
        _waiters += 1
        try:
            while _busy:
                _condition.wait()
            _busy = True
        finally:
            _waiters -= 1
    try:
        yield
    finally:
        with _condition:
            _busy = False
            _condition.notify_all()
