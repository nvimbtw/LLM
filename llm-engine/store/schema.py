"""What the LLM engine is the authority on: models, training, vocabularies and execution scheduling.

Deliberately not here: anything about a person. `owner_id` is an opaque id
minted by the backend and never validated — the LLM engine scopes by it and has no
way to resolve it, which is what keeps the two databases independent.
"""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Postgres in production, SQLite in tests — the tests need no server and the
# schema uses nothing dialect-specific beyond these two. `Uuid` is SQLAlchemy's
# own cross-dialect type: a native uuid column on Postgres, CHAR elsewhere, and
# a Python uuid.UUID on both sides of the wire.
Json = JSON().with_variant(JSONB(), "postgresql")
Key = Uuid(as_uuid=True)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class RunKind(str, enum.Enum):
    train = "train"
    benchmark = "benchmark"
    #: Learning a vocabulary. The one kind that does not touch the GPU — it is a
    #: run because it takes minutes and wants the queue's frames and cancel, and
    #: because building one while the device trains would fight it for the CPU.
    tokenize = "tokenize"
    #: Reinforcement learning against a gym. A `train` run in every way that
    #: matters — it writes checkpoints, produces a model, and can be chained
    #: from — and a kind of its own because what it *reports* is a pass rate
    #: rather than a loss curve, and nothing that reads a run's frames can tell
    #: those apart from the numbers alone.
    rl = "rl"


class RunStatus(str, enum.Enum):
    #: Authored but not runnable yet — it consumes a model another run produces.
    waiting = "waiting"
    queued = "queued"
    running = "running"
    #: A running job can only stop at a step boundary, so there is a real
    #: interval between asking and it noticing.
    cancelling = "cancelling"
    done = "done"
    cancelled = "cancelled"
    failed = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (RunStatus.done, RunStatus.cancelled, RunStatus.failed)


class Model(Base):
    """A trained model. The row is the index; the weights are on disk under
    `artifacts/models/<id>/`, which is why the id and not the name is stable.

    A training run's intermediate checkpoints are rows here too, laid out on
    disk exactly like this one: that is what makes chatting with a checkpoint,
    benchmarking it or continuing from it fall out of the code that already
    exists rather than each needing a second path. `checkpoint_index IS NOT
    NULL` is the whole definition of "this row is an intermediate" — there is
    deliberately no boolean beside it that could disagree.
    """

    __tablename__ = "models"

    id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    family: Mapped[str | None] = mapped_column(String(120))
    #: The TrainConfig the run was started with, stored opaquely.
    parameters: Mapped[dict | None] = mapped_column(Json)
    context: Mapped[int | None] = mapped_column(Integer)
    #: Private until published, so a half-trained model is not offered in chat.
    published: Mapped[bool] = mapped_column(Boolean, default=False)
    #: The run that produced it, carried by a checkpoint and by the finished
    #: model alike so a run's whole output groups by one value. Not a foreign
    #: key: execution rows are prunable and the
    #: grouping has to outlive them.
    run_id: Mapped[uuid.UUID | None] = mapped_column(Key, index=True)
    #: The finished model this is an intermediate of. Also not a foreign key,
    #: but for a different reason: the parent row does not exist yet when the
    #: first checkpoint of a run lands. Null on a finished model — it is the
    #: parent.
    parent_id: Mapped[uuid.UUID | None] = mapped_column(Key, index=True)
    #: The global step it was saved at. Null means this row is a finished model.
    checkpoint_step: Mapped[int | None] = mapped_column(Integer)
    #: 1-based ordinal within the run, and the `#n` its name ends with.
    checkpoint_index: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)




class Tokenizer(Base):
    """A built vocabulary. The row is the index; the vocabulary is on disk under
    `artifacts/tokenizers/<id>/`, which is why the id and not the name is stable.

    No foreign key from `models`: a model carries its own copy of the vocabulary
    it was trained with, and must survive the named build being deleted.
    """

    __tablename__ = "tokenizers"

    id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    vocab_size: Mapped[int | None] = mapped_column(Integer)
    merges: Mapped[int | None] = mapped_column(Integer)
    max_vocab: Mapped[int | None] = mapped_column(Integer)
    #: A digest of the saved vocabulary. What a packed corpus is keyed on, since
    #: rebuilding under one id renumbers every token — so the id alone cannot
    #: answer "is this the same vocabulary".
    fingerprint: Mapped[str | None] = mapped_column(String(64))
    #: The [[name, cap], …] it was built from, as resolved at build time.
    datasets: Mapped[list | None] = mapped_column(Json)
    #: Derived from a training run's own selection rather than asked for by name.
    #: Listed apart from the built-on-purpose ones, which are what a composer offers.
    implicit: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Run(Base):
    """A unit of work for the device.

    One device, one line, one runner. There is no lane and no priority band:
    both existed for kinds of work that no longer exist, and everything left
    here contends for the same GPU.
    """

    __tablename__ = "execution_jobs"

    id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True, default=uuid.uuid4)
    kind: Mapped[RunKind] = mapped_column(Enum(RunKind, name="run_kind"))
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, name="run_status"), default=RunStatus.queued, index=True
    )
    #: What to call this in a list. Denormalized for the same reason
    #: `Benchmark.model_name` is.
    label: Mapped[str] = mapped_column(String(200))
    #: Ordering within the queue. A new run goes in above the highest; a drag
    #: writes the midpoint of its new neighbours, so reordering touches one row.
    rank: Mapped[float] = mapped_column(Float, default=0.0)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str | None] = mapped_column(String(120))
    error: Mapped[str | None] = mapped_column(Text)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    #: The run whose model this one consumes. A real reference, unlike `ref_id`,
    #: because it always names another row here.
    depends_on: Mapped[uuid.UUID | None] = mapped_column(
        Key, ForeignKey("execution_jobs.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


    training: Mapped["TrainingRun | None"] = relationship(cascade="all, delete-orphan", lazy="joined", uselist=False)
    tokenizer_build: Mapped["TokenizerBuild | None"] = relationship(cascade="all, delete-orphan", lazy="joined", uselist=False)
    benchmark_dispatch: Mapped["BenchmarkDispatch | None"] = relationship(cascade="all, delete-orphan", lazy="joined", uselist=False)

    def _domain(self):
        if self.kind in (RunKind.train, RunKind.rl):
            if self.training is None:
                self.training = TrainingRun(training_type="rl" if self.kind == RunKind.rl else "pretrain")
            return self.training
        if self.kind == RunKind.tokenize:
            if self.tokenizer_build is None:
                self.tokenizer_build = TokenizerBuild()
            return self.tokenizer_build
        if self.kind == RunKind.benchmark:
            if self.benchmark_dispatch is None:
                self.benchmark_dispatch = BenchmarkDispatch()
            return self.benchmark_dispatch
        raise ValueError(f"unsupported execution kind {self.kind}")

    @property
    def config(self):
        return self._domain().config or {}

    @config.setter
    def config(self, value):
        domain = self._domain()
        domain.config = value
        if isinstance(domain, TrainingRun):
            domain.training_type = "rl" if self.kind == RunKind.rl else "sft" if value.get("base_model") else "pretrain"
            domain.target_model_id = _uuid(value.get("model_name"))
        elif isinstance(domain, TokenizerBuild):
            domain.target_tokenizer_id = _uuid(value.get("tokenizer_id"))

    @property
    def result(self):
        return self._domain().result

    @result.setter
    def result(self, value):
        self._domain().result = value

    @property
    def ref_id(self):
        domain = self._domain()
        if isinstance(domain, TrainingRun):
            return domain.model_id
        if isinstance(domain, TokenizerBuild):
            return domain.tokenizer_id
        return domain.benchmark_id

    @ref_id.setter
    def ref_id(self, value):
        domain = self._domain()
        if isinstance(domain, TrainingRun):
            domain.model_id = value
        elif isinstance(domain, TokenizerBuild):
            domain.tokenizer_id = value
        else:
            domain.benchmark_id = value


def _uuid(value):
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None  # Historical runs may predate reserved output UUIDs.


class TrainingRun(Base):
    """Persist training inputs and outputs with the migrated training-type check."""

    __tablename__ = "training_runs"
    __table_args__ = (
        CheckConstraint(
            "training_type IN ('pretrain','sft','rl')",
            name="training_runs_training_type_check",
        ),
    )
    job_id: Mapped[uuid.UUID] = mapped_column(Key, ForeignKey("execution_jobs.id", ondelete="CASCADE"), primary_key=True)
    training_type: Mapped[str] = mapped_column(String(12))
    config: Mapped[dict] = mapped_column(Json, default=dict)
    result: Mapped[dict | None] = mapped_column(Json)
    target_model_id: Mapped[uuid.UUID | None] = mapped_column(Key)
    model_id: Mapped[uuid.UUID | None] = mapped_column(Key, ForeignKey("models.id", ondelete="SET NULL"))


class TokenizerBuild(Base):
    __tablename__ = "tokenizer_builds"
    job_id: Mapped[uuid.UUID] = mapped_column(Key, ForeignKey("execution_jobs.id", ondelete="CASCADE"), primary_key=True)
    config: Mapped[dict] = mapped_column(Json, default=dict)
    result: Mapped[dict | None] = mapped_column(Json)
    target_tokenizer_id: Mapped[uuid.UUID | None] = mapped_column(Key)
    tokenizer_id: Mapped[uuid.UUID | None] = mapped_column(Key, ForeignKey("tokenizers.id", ondelete="SET NULL"))


class BenchmarkDispatch(Base):
    """Queue input and remote reference; scoring/history belongs to bench-engine."""
    __tablename__ = "benchmark_dispatches"
    job_id: Mapped[uuid.UUID] = mapped_column(Key, ForeignKey("execution_jobs.id", ondelete="CASCADE"), primary_key=True)
    config: Mapped[dict] = mapped_column(Json, default=dict)
    benchmark_id: Mapped[uuid.UUID | None] = mapped_column(Key)

    @property
    def result(self):
        return {"benchmark_id": str(self.benchmark_id)} if self.benchmark_id else None

    @result.setter
    def result(self, value):
        if value and value.get("benchmark_id"):
            self.benchmark_id = _uuid(value["benchmark_id"])
