import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from store.schema import Base, EpisodeStatus, GymEpisode, GymRun, GymStep, RunStatus

__all__ = [
    "Base",
    "EpisodeStatus",
    "GymEpisode",
    "GymRun",
    "GymStep",
    "RunStatus",
    "configure",
    "session",
    "url",
]

DEFAULT_URL = "postgresql+psycopg://localhost/slate_gyms"

_sessions: sessionmaker[Session] | None = None


def url() -> str:
    return os.environ.get("GYMS_DATABASE_URL") or DEFAULT_URL


def configure(database_url: str | None = None, *, create: bool = False) -> sessionmaker[Session]:
    global _sessions
    resolved = database_url or url()
    engine = create_engine(resolved, pool_pre_ping=True, future=True, **_options(resolved))
    if create:
        Base.metadata.create_all(engine)
    _sessions = sessionmaker(engine, expire_on_commit=False)
    return _sessions


def _options(database_url: str) -> dict:
    if not database_url.startswith("sqlite") or ":memory:" not in database_url:
        return {}
    return {"poolclass": StaticPool, "connect_args": {"check_same_thread": False}}


@contextmanager
def session() -> Iterator[Session]:
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
