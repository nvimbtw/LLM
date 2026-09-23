"""What the LLM engine is the authority on: models, conversations, runs, scores.

Deliberately not here: anything about a person. `owner_id` is an opaque id
minted by the backend and never validated — the LLM engine scopes by it and has no
way to resolve it, which is what keeps the two databases independent.
"""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
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
    chat = "chat"
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


class Suite(str, enum.Enum):
    language = "language"
    markup = "markup"
    assistant = "assistant"


class BenchmarkStatus(str, enum.Enum):
    running = "running"
    done = "done"
    failed = "failed"


class Role(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    #: A tool's answer to a call the previous assistant message made. Stored as
    #: a message so the history a later turn is built from is whole — the model
    #: has to see what it was told, not just that it asked.
    tool = "tool"


class Feedback(str, enum.Enum):
    up = "up"
    down = "down"


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
    #: key, for the reason `Run.ref_id` is not one: run rows are prunable and the
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

    benchmarks: Mapped[list["Benchmark"]] = relationship(
        back_populates="model", cascade="all, delete-orphan"
    )


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


class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True, default=uuid.uuid4)
    #: A `users.id` in the backend's database. No foreign key — the row lives in
    #: another database, and a chat whose owner has gone is invisible, not broken.
    owner_id: Mapped[uuid.UUID] = mapped_column(Key, index=True)
    title: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(Base):
    """One turn.

    `role` is stored rather than derived from `model_id`: deleting a model nulls
    the credit, and a reply must not turn into a question because the model that
    wrote it is gone.
    """

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True, default=uuid.uuid4)
    chat_id: Mapped[uuid.UUID] = mapped_column(
        Key, ForeignKey("chats.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[Role] = mapped_column(Enum(Role, name="role"))
    #: Which model wrote it, while that model still exists.
    model_id: Mapped[uuid.UUID | None] = mapped_column(
        Key, ForeignKey("models.id", ondelete="SET NULL")
    )
    content: Mapped[str] = mapped_column(Text)
    #: Length under the model's own tokenizer, counted once at write time so a
    #: chat's cost can be summed without re-tokenizing it.
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    #: Null is "not rated", which is almost every row and has to stay distinct
    #: from a deliberate rating.
    feedback: Mapped[Feedback | None] = mapped_column(Enum(Feedback, name="feedback"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    chat: Mapped[Chat] = relationship(back_populates="messages")
    #: Read-only: what a reply is labelled with. Deleting a model nulls the FK,
    #: and the turn keeps its role.
    model: Mapped["Model | None"] = relationship(lazy="joined")
    attachments: Mapped[list["Attachment"]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        order_by="Attachment.created_at",
    )


class Attachment(Base):
    """A file a person put in the composer.

    Stored and shown back in the history, and read by nothing: the models served
    here take text and only text, so an attachment is content the *conversation*
    carries rather than content the *model* is given. Nothing under `llm/` may
    touch this table — see
    `docs/decisions/attachments-are-stored-not-read.md`.

    The bytes are a column rather than a file under `data/`, which is derived
    artifacts and nothing else; an upload is derived from nothing. It also makes
    forgetting one a cascade rather than a directory walk that has to not go
    wrong.

    `message_id` is null between the upload and the turn it is sent with — a
    file is picked before there is a message to hang it on, and on a new chat
    before there is even a chat. So `owner_id` is what scopes a row in both
    states, and it is the column that answers "may this caller see this file".
    """

    __tablename__ = "attachments"

    id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True, default=uuid.uuid4)
    #: A `users.id` in the backend's database, carried for the same reason
    #: `Chat.owner_id` is, and never resolved here.
    owner_id: Mapped[uuid.UUID] = mapped_column(Key, index=True)
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        Key, ForeignKey("messages.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(255))
    #: What the browser said it was. Believed for display and never for parsing,
    #: because nothing here parses it.
    media_type: Mapped[str] = mapped_column(String(160))
    #: Denormalised from `content` so a listing can report a size without
    #: loading the bytes to measure them.
    size: Mapped[int] = mapped_column(Integer)
    #: Deferred: every history read walks these rows for their metadata and
    #: exactly one route ever wants the bytes. Without it, opening a chat with
    #: ten images in it loads ten images to render ten filenames.
    content: Mapped[bytes] = mapped_column(LargeBinary, deferred=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    message: Mapped["Message | None"] = relationship(back_populates="attachments")


class Benchmark(Base):
    """One suite's verdict on one model.

    The three suites share a range and nothing else — never compared, never
    averaged — so a row carries which suite produced it and at what version.
    Two scores are only comparable when both match.
    """

    __tablename__ = "benchmarks"

    id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID] = mapped_column(
        Key, ForeignKey("models.id", ondelete="CASCADE"), index=True
    )
    #: The model's name when the run happened. Denormalized because a rename
    #: must not retroactively relabel the score it earned under the old one.
    model_name: Mapped[str] = mapped_column(String(120))
    suite: Mapped[Suite] = mapped_column(Enum(Suite, name="suite"))
    suite_version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[BenchmarkStatus] = mapped_column(
        Enum(BenchmarkStatus, name="benchmark_status"), default=BenchmarkStatus.running
    )
    #: Lifted out of `report` so the history list and the trend chart need not
    #: parse a JSON blob per row.
    score: Mapped[float | None] = mapped_column(Float)
    report: Mapped[dict | None] = mapped_column(Json)
    #: What the measurement cost, in the unit the model works in.
    eval_tokens: Mapped[int | None] = mapped_column(BigInteger)
    train_tokens: Mapped[int | None] = mapped_column(BigInteger)
    eval_conversations: Mapped[int | None] = mapped_column(BigInteger)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    model: Mapped[Model] = relationship(back_populates="benchmarks")


class Run(Base):
    """A unit of work for the device.

    One device, one line, one runner. There is no lane and no priority band:
    both existed for kinds of work that no longer exist, and everything left
    here contends for the same GPU.
    """

    __tablename__ = "runs"

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
    #: Everything needed to run it, and to run it again from a retry.
    config: Mapped[dict] = mapped_column(Json, default=dict)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str | None] = mapped_column(String(120))
    #: What it produced: a `models.id`, a `benchmarks.id`, or a `messages.id`.
    #: Not a foreign key — it points into whichever table the kind writes.
    ref_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    #: A summary for kinds whose output has nowhere else to live.
    result: Mapped[dict | None] = mapped_column(Json)
    error: Mapped[str | None] = mapped_column(Text)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    #: The run whose model this one consumes. A real reference, unlike `ref_id`,
    #: because it always names another row here.
    depends_on: Mapped[uuid.UUID | None] = mapped_column(
        Key, ForeignKey("runs.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class QueueState(Base):
    """Whether the runner is draining the queue.

    A row rather than a flag in memory: pausing exists so the machine can be
    left alone, and a restart in the middle of that must not quietly resume.
    """

    __tablename__ = "queue_state"

    only_row: Mapped[bool] = mapped_column(Boolean, primary_key=True, default=True)
    paused: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
