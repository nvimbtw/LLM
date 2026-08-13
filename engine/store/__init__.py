"""The engine's own database.

Sync rather than async on purpose: a run's worker thread writes its own
progress, and reaching an async session from a thread means bouncing every
write through the event loop for no gain at one process and one device.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from store.schema import (
    Base,
    Benchmark,
    BenchmarkStatus,
    Chat,
    Feedback,
    Message,
    Model,
    QueueState,
    Run,
    RunKind,
    RunStatus,
    Suite,
)

__all__ = [
    "Base", "Benchmark", "BenchmarkStatus", "Chat", "Feedback", "Message", "Model",
    "QueueState", "Run", "RunKind", "RunStatus", "Suite",
    "configure", "session", "url",
]

DEFAULT_URL = "postgresql+psycopg://localhost/slate_engine"

_sessions: sessionmaker[Session] | None = None


def url() -> str:
    return os.environ.get("ENGINE_DATABASE_URL") or DEFAULT_URL


def configure(database_url: str | None = None, *, create: bool = False) -> sessionmaker[Session]:
    """Bind the session factory. Idempotent — calling it again rebinds."""
    global _sessions
    resolved = database_url or url()
    engine = create_engine(resolved, pool_pre_ping=True, future=True, **_options(resolved))
    if create:
        Base.metadata.create_all(engine)
    _sessions = sessionmaker(engine, expire_on_commit=False)
    return _sessions


def _options(database_url: str) -> dict:
    """An in-memory SQLite database belongs to its connection, so a run's worker
    thread opening a second one finds an empty schema. One shared connection
    makes the test database behave like the server the code actually runs on."""
    if not database_url.startswith("sqlite") or ":memory:" not in database_url:
        return {}
    return {"poolclass": StaticPool, "connect_args": {"check_same_thread": False}}


@contextmanager
def session() -> Iterator[Session]:
    """A transaction. Commits on a clean exit, rolls back on an exception."""
    if _sessions is None:
        configure()
    assert _sessions is not None
    db = _sessions()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
