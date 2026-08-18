"""What the server's own logging says, and what it swallows.

Uvicorn logs one access line per request, which is the right default for a
server whose traffic is people. This one's traffic is mostly a browser asking
the same question on a timer, and a terminal that scrolls a screen of `GET
/runs` a minute is a terminal nobody reads — including when a real error goes
past.
"""

import logging

#: Paths whose successful GETs are a client polling, not an event. Matched
#: whole, so `/runs/{id}` — a person opening a job — still shows up.
POLLED_PATHS = frozenset({"/runs", "/"})

#: Uvicorn builds its access line from a fixed 5-tuple of args; anything else on
#: the access logger is not an access record and is left alone.
_ACCESS_ARITY = 5


class _QuietPolls(logging.Filter):
    """Drops access lines for polls that succeeded. Failures always survive."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) != _ACCESS_ARITY:
            return True
        _, method, path, _, status = args
        if method != "GET" or int(status) >= 400:
            return True
        return path.split("?", 1)[0] not in POLLED_PATHS


def quieten() -> None:
    """Silences successful polls on uvicorn's access log. Idempotent.

    Called at import of `main`, so it holds however the server was started —
    spawned by the backend or `uvicorn main:app --reload` by hand — rather than
    only under a log config someone remembered to pass.
    """
    access = logging.getLogger("uvicorn.access")
    if any(isinstance(f, _QuietPolls) for f in access.filters):
        return
    access.addFilter(_QuietPolls())
