"""Benchmark records; execution queue status belongs to the LLM scheduler."""
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from sqlalchemy import DateTime, Float, Integer, BigInteger, String, Text, Uuid, create_engine, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


def now():
    return datetime.now(timezone.utc)


class Benchmark(Base):
    __tablename__ = 'benchmarks'
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID] = mapped_column(Uuid, index=True)
    model_name: Mapped[str] = mapped_column(String(120))
    suite: Mapped[str] = mapped_column(String(32), default='language')
    suite_version: Mapped[int] = mapped_column(Integer, default=2)
    status: Mapped[str] = mapped_column(String(32), default='running')
    config: Mapped[dict | None] = mapped_column(JSON().with_variant(JSONB(),'postgresql'))
    score: Mapped[float | None] = mapped_column(Float)
    report: Mapped[dict | None] = mapped_column(JSON().with_variant(JSONB(),'postgresql'))
    eval_tokens: Mapped[int | None] = mapped_column(BigInteger)
    train_tokens: Mapped[int | None] = mapped_column(BigInteger)
    eval_conversations: Mapped[int | None] = mapped_column(BigInteger)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


def url():
    return os.environ.get('BENCH_ENGINE_DATABASE_URL','postgresql+psycopg://localhost/slate_bench')


_sessions = None


def configure():
    global _sessions
    _sessions = sessionmaker(create_engine(url(),pool_pre_ping=True),expire_on_commit=False)


@contextmanager
def session():
    if _sessions is None:
        configure()
    with _sessions.begin() as db:
        yield db


def describe(row, full=False):
    fields = ('model_name','suite','suite_version','status','score','eval_tokens','eval_conversations','error')
    result = {key:getattr(row,key) for key in fields}
    result.update(benchmark_id=str(row.id),model_id=str(row.model_id),
                  created_at=row.created_at.isoformat(),
                  completed_at=row.completed_at.isoformat() if row.completed_at else None)
    if full:
        result['report'] = row.report
    return result
