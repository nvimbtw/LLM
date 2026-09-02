"""What the data engine is the authority on: datasets, their documents, and the
runs that produce them.

Content lives here rather than on disk. The engine's `data/` tree grew large
enough to slow every tool that walks the repo, and a corpus that is only ever
read by page, sampled by ordinal and filtered by predicate is a table's job
anyway. The cost is real and is paid deliberately: a pull is capped, inserts are
batched, and nothing reads a whole dataset into memory.

Deliberately not here: anything about a person, and anything the LLM engine
owns. `owner_id` is an opaque id minted by the backend and never validated —
the same arrangement that keeps the engine's database independent of the
backend's.
"""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Postgres in production, SQLite in tests — the same two aliases the engine
# uses, for the same reason.
#
# `none_as_null` is not cosmetic: without it a Python `None` is stored as the
# JSON value `null`, which is not SQL NULL, so `messages IS NULL` never holds and
# the content/messages constraint below rejects every row ever written.
Json = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")
Key = Uuid(as_uuid=True)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class DatasetKind(str, enum.Enum):
    #: Prose, one document per row, destined for next-token training.
    pretrain = "pretrain"
    #: Conversations, one exchange per row.
    sft = "sft"


class DatasetOrigin(str, enum.Enum):
    huggingface = "huggingface"
    #: Produced by running a pipeline over another dataset here.
    derived = "derived"
    #: Captured as a side effect of a tool call — a `web-search` today —
    #: rather than pulled in bulk or processed from something that was. See
    #: `store.queries.cache_dataset`.
    search = "search"
    #: Expert demonstrations played through a gym's sandbox and written here by
    #: the backend. Not pulled, not derived from anything in this database, and
    #: not a recording of a model — the gym's own generator holds the ground
    #: truth, so it can demonstrate itself. `docs/gyms/demonstrations.md`.
    gym = "gym"


class DatasetStatus(str, enum.Enum):
    #: Rows are still arriving. Selectable nowhere until it is `ready`, because a
    #: half-pulled corpus trains as silently as a whole one.
    pulling = "pulling"
    ready = "ready"
    failed = "failed"


class RunKind(str, enum.Enum):
    pull = "pull"
    process = "process"


class RunStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    #: A running job can only stop between batches, so there is a real interval
    #: between asking and it noticing.
    cancelling = "cancelling"
    done = "done"
    cancelled = "cancelled"
    failed = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (RunStatus.done, RunStatus.cancelled, RunStatus.failed)


class Dataset(Base):
    """A named body of documents, either pulled from the Hub or derived from one
    that was.

    A derived dataset records the pipeline that produced it rather than pointing
    at a recipe row: the processors are code, they are versioned with the repo,
    and a saved recipe that drifted from the run it describes would be a lie
    about provenance.
    """

    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    kind: Mapped[DatasetKind] = mapped_column(Enum(DatasetKind, name="dataset_kind"))
    origin: Mapped[DatasetOrigin] = mapped_column(Enum(DatasetOrigin, name="dataset_origin"))
    status: Mapped[DatasetStatus] = mapped_column(
        Enum(DatasetStatus, name="dataset_status"), default=DatasetStatus.pulling, index=True
    )

    #: Where it came from on the Hub, as asked for. Kept on derived rows too, so
    #: the trail back to a source repo survives without walking every parent.
    source_repo: Mapped[str | None] = mapped_column(String(200))
    source_config: Mapped[str | None] = mapped_column(String(120))
    source_split: Mapped[str | None] = mapped_column(String(120))

    #: The dataset this one was processed from. `SET NULL` rather than cascade:
    #: deleting a source must not silently take its cleaned descendants with it.
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        Key, ForeignKey("datasets.id", ondelete="SET NULL"), index=True
    )
    #: The ordered `[{processor, params}, …]` that produced it, resolved at run
    #: time. Null on a pulled dataset.
    steps: Mapped[list | None] = mapped_column(Json)

    #: Counted at write time. A `SELECT count(*)` over a corpus-sized table is
    #: the one query a list page must never make.
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    byte_size: Mapped[int] = mapped_column(BigInteger, default=0)

    owner_id: Mapped[uuid.UUID | None] = mapped_column(Key, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    documents: Mapped[list["Document"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class Document(Base):
    """One row of a dataset: a passage for `pretrain`, an exchange for `sft`.

    `ordinal` rather than `created_at` is what orders a dataset, and it is what
    makes a preview honest: "the first twenty documents" must mean the same
    twenty every time the user nudges a parameter, or the before/after they are
    comparing is two different samples.
    """

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("dataset_id", "ordinal", name="uq_documents_dataset_ordinal"),
        # A pretrain row carries prose and an SFT row carries turns; a row
        # carrying both, or neither, is a bug in whatever wrote it.
        CheckConstraint(
            "(content IS NULL) <> (messages IS NULL)",
            name="ck_documents_content_xor_messages",
        ),
        # What a cache lookup filters on. Composite rather than a bare index on
        # `cache_key` alone because every cache row lives in one dataset today,
        # but nothing here should assume that stays true.
        Index("ix_documents_dataset_id_cache_key", "dataset_id", "cache_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        Key, ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    content: Mapped[str | None] = mapped_column(Text)
    #: `[{"role": …, "content": …}, …]`, in the order they were said.
    messages: Mapped[list | None] = mapped_column(Json)
    #: Whatever Hub columns were worth keeping — url, score, language. Carried
    #: through processing untouched so a filter can be written against it later.
    meta: Mapped[dict | None] = mapped_column(Json)
    byte_size: Mapped[int] = mapped_column(Integer, default=0)
    #: When this row was written. Every document has one, not only a cached
    #: search's — nothing here carried a timestamp before, and every other
    #: table in this schema already does; the cache is just the first reader
    #: that needs it, to answer "how old is this."
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    #: The normalized key a cache write is found by (a web-search query,
    #: today). Null on anything pulled from the Hub or produced by a
    #: pipeline — those are found by dataset and ordinal, never by content.
    cache_key: Mapped[str | None] = mapped_column(String(500))

    dataset: Mapped[Dataset] = relationship(back_populates="documents")


class Run(Base):
    """A pull or a processing pass.

    Same shape as the engine's queue and for the same reasons — it takes minutes,
    it wants frames and a cancel — but a separate line: nothing here touches the
    GPU, so a pull has no business queueing behind a training run.
    """

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True, default=uuid.uuid4)
    kind: Mapped[RunKind] = mapped_column(Enum(RunKind, name="run_kind"))
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, name="run_status"), default=RunStatus.queued, index=True
    )
    label: Mapped[str] = mapped_column(String(200))
    config: Mapped[dict] = mapped_column(Json, default=dict)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str | None] = mapped_column(String(120))
    #: The `datasets.id` it writes. Not a foreign key: the run outlives the
    #: dataset it produced, and a deleted dataset must not erase its history.
    ref_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    result: Mapped[dict | None] = mapped_column(Json)
    error: Mapped[str | None] = mapped_column(Text)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(Key, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
