import enum
import uuid
from datetime import datetime, timezone
from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Enum, Float, ForeignKey, String, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
Json = JSON().with_variant(JSONB(), "postgresql")
Key = Uuid(as_uuid=True)
def _now() -> datetime:
    return datetime.now(timezone.utc)
class Base(DeclarativeBase):
    pass
class RunKind(str, enum.Enum):
    benchmark = "benchmark"
class RunStatus(str, enum.Enum):
    waiting = "waiting"
    queued = "queued"
    running = "running"
    cancelling = "cancelling"
    done = "done"
    cancelled = "cancelled"
    failed = "failed"
    @property
    def is_terminal(self) -> bool:
        return self in (RunStatus.done, RunStatus.cancelled, RunStatus.failed)
class Suite(str, enum.Enum):
    language = "language"
    language_v3 = "language_v3"
    language_v3_hyphen = "language-v3"
class BenchmarkStatus(str, enum.Enum):
    running = "running"
    done = "done"
    failed = "failed"
class Benchmark(Base):
    __tablename__ = "benchmarks"
    id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID] = mapped_column(Key, index=True)
    model_name: Mapped[str] = mapped_column(String(120))
    suite: Mapped[str] = mapped_column(String(20))
    suite_version: Mapped[int] = mapped_column(default=1)
    status: Mapped[BenchmarkStatus] = mapped_column(Enum(BenchmarkStatus, name="benchmark_status"), default=BenchmarkStatus.running)
    score: Mapped[float | None] = mapped_column(Float)
    report: Mapped[dict | None] = mapped_column(Json)
    eval_tokens: Mapped[int | None] = mapped_column(BigInteger)
    eval_conversations: Mapped[int | None] = mapped_column(BigInteger)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
class Run(Base):
    __tablename__ = "runs"
    id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True, default=uuid.uuid4)
    kind: Mapped[RunKind] = mapped_column(Enum(RunKind, name="run_kind"))
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus, name="run_status"), default=RunStatus.queued, index=True)
    label: Mapped[str] = mapped_column(String(200))
    rank: Mapped[float] = mapped_column(Float, default=0.0)
    config: Mapped[dict] = mapped_column(Json, default=dict)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str | None] = mapped_column(String(120))
    ref_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    result: Mapped[dict | None] = mapped_column(Json)
    error: Mapped[str | None] = mapped_column(Text)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    depends_on: Mapped[uuid.UUID | None] = mapped_column(Key, ForeignKey("runs.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
class QueueState(Base):
    __tablename__ = "queue_state"
    only_row: Mapped[bool] = mapped_column(Boolean, primary_key=True, default=True)
    paused: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
