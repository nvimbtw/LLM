"""Reads and writes, one function per question.

Everything that scopes by owner takes it as an argument rather than reading it
from anywhere — this service cannot resolve an owner, so forgetting to pass one
has to be a type error rather than a silent full-table read.

Nothing here returns a whole dataset. A corpus-sized table is read by page or
streamed; the functions that touch `documents` all take a bound.
"""

import json
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from store.schema import (
    Dataset,
    DatasetKind,
    DatasetOrigin,
    DatasetStatus,
    Document,
    Run,
    RunKind,
    RunStatus,
)

#: How many rows a streaming read pulls per round trip. Large enough that a
#: million-row export is not a million round trips, small enough that one batch
#: of long documents is not a surprise allocation.
STREAM_BATCH = 500

#: The one dataset every cached tool result lands in. Reserved rather than
#: chosen per-call, so every caller finds the same rows.
CACHE_DATASET_NAME = "web-search-cache"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def list_datasets(db: Session, owner_id: uuid.UUID | None = None) -> list[Dataset]:
    query = select(Dataset).order_by(Dataset.created_at.desc())
    if owner_id is not None:
        query = query.where(Dataset.owner_id == owner_id)
    return list(db.scalars(query))


def dataset(db: Session, dataset_id: uuid.UUID) -> Dataset | None:
    return db.get(Dataset, dataset_id)


def dataset_by_name(db: Session, name: str) -> Dataset | None:
    return db.scalar(select(Dataset).where(Dataset.name == name))


def create_dataset(
    db: Session,
    *,
    name: str,
    kind: DatasetKind,
    origin: DatasetOrigin,
    owner_id: uuid.UUID | None,
    description: str | None = None,
    source_repo: str | None = None,
    source_config: str | None = None,
    source_split: str | None = None,
    parent_id: uuid.UUID | None = None,
    steps: list | None = None,
    #: `pulling` fits a Hub pull, which fills over minutes and is not safe to
    #: select from until it finishes. Nothing else in this module produces
    #: that half-written state, so a caller that knows its rows are already
    #: complete (the cache) may say so up front.
    status: DatasetStatus = DatasetStatus.pulling,
) -> Dataset:
    row = Dataset(
        name=name,
        kind=kind,
        origin=origin,
        owner_id=owner_id,
        description=description,
        source_repo=source_repo,
        source_config=source_config,
        source_split=source_split,
        parent_id=parent_id,
        steps=steps,
        status=status,
    )
    db.add(row)
    db.flush()
    return row


def finish_dataset(db: Session, dataset_id: uuid.UUID, status: DatasetStatus) -> None:
    """Publish or condemn a dataset, recounting what actually landed.

    The counters are recomputed here rather than accumulated by the writer: a
    pull that was cancelled halfway through must still report the rows it kept,
    and a writer that crashed cannot be trusted to have added up its own work.
    """
    row = db.get(Dataset, dataset_id)
    if row is None:
        return
    totals = db.execute(
        select(func.count(Document.id), func.coalesce(func.sum(Document.byte_size), 0)).where(
            Document.dataset_id == dataset_id
        )
    ).one()
    row.document_count, row.byte_size = int(totals[0]), int(totals[1])
    row.status = status
    row.updated_at = _now()


def delete_dataset(db: Session, dataset_id: uuid.UUID) -> bool:
    row = db.get(Dataset, dataset_id)
    if row is None:
        return False
    db.delete(row)
    return True


def add_documents(db: Session, dataset_id: uuid.UUID, start: int, rows: list[dict]) -> int:
    """Append a batch, numbering from `start`. Returns how many were written.

    A bulk insert rather than ORM objects: a pull writes hundreds of thousands of
    these and the identity map is pure overhead when nothing reads them back.
    """
    if not rows:
        return 0
    payload = [
        {
            "id": uuid.uuid4(),
            "dataset_id": dataset_id,
            "ordinal": start + offset,
            "content": row.get("content"),
            "messages": row.get("messages"),
            "meta": row.get("meta"),
            "byte_size": row.get("byte_size", 0),
        }
        for offset, row in enumerate(rows)
    ]
    db.execute(Document.__table__.insert(), payload)
    return len(payload)


def documents(db: Session, dataset_id: uuid.UUID, *, offset: int, limit: int) -> list[Document]:
    return list(
        db.scalars(
            select(Document)
            .where(Document.dataset_id == dataset_id)
            .order_by(Document.ordinal)
            .offset(offset)
            .limit(limit)
        )
    )


def stream_documents(db: Session, dataset_id: uuid.UUID) -> Iterator[Document]:
    """Every document, in order, without holding them all.

    `yield_per` keeps the result set on the server side, which is the difference
    between exporting a large corpus and running the process out of memory.
    """
    query = (
        select(Document).where(Document.dataset_id == dataset_id).order_by(Document.ordinal)
    )
    yield from db.scalars(query).yield_per(STREAM_BATCH)


def clear_documents(db: Session, dataset_id: uuid.UUID) -> None:
    """Drop a dataset's rows without dropping the dataset — what a retried pull
    needs so a partial first attempt does not double every document."""
    db.execute(delete(Document).where(Document.dataset_id == dataset_id))


def cache_dataset(db: Session) -> Dataset:
    """The one dataset every cached tool result lands in, created the first
    time anything is cached rather than by a migration — a fresh checkout has
    made no searches yet, and a migration writing a name is a stranger
    decision than the code that will actually read it making one.

    Immediately `ready`: unlike a Hub pull there is no batch to wait out, so
    there is no interval where it would be true to call this half-written.
    """
    row = dataset_by_name(db, CACHE_DATASET_NAME)
    if row is not None:
        return row
    try:
        with db.begin_nested():
            return create_dataset(
                db,
                name=CACHE_DATASET_NAME,
                kind=DatasetKind.pretrain,
                origin=DatasetOrigin.search,
                owner_id=None,
                description="Web search results, cached by the web-search tool.",
                status=DatasetStatus.ready,
            )
    except IntegrityError:
        # Lost a race with another writer creating the same row between the
        # lookup above and this insert — its commit is the one that landed.
        row = dataset_by_name(db, CACHE_DATASET_NAME)
        if row is None:
            raise
        return row


def latest_by_key(db: Session, dataset_id: uuid.UUID, cache_key: str) -> Document | None:
    """The freshest document under `cache_key` in this dataset, or None.

    A re-fetch appends rather than replaces — the same shape a re-pulled Hub
    dataset already has, "holding more documents under the same id" (see
    `docs/engine/README.md`) — so the newest row by `created_at` is the one
    that answers "what do we currently believe," and nothing is ever
    overwritten in place.
    """
    return db.scalar(
        select(Document)
        .where(Document.dataset_id == dataset_id, Document.cache_key == cache_key)
        .order_by(Document.created_at.desc())
        .limit(1)
    )


def add_document(
    db: Session, dataset_id: uuid.UUID, *, cache_key: str, content: str, meta: dict | None = None
) -> Document:
    """Append one row outside the batch `Run` path — a live tool call landing a
    single result, not a pull filling hundreds of thousands.

    The dataset row is locked for the write's duration, which is what lets
    concurrent callers land in the same dataset without racing `ordinal` or
    leaving `document_count`/`byte_size` to be recomputed from scratch the way
    `finish_dataset` does for a batch.
    """
    dataset = db.get(Dataset, dataset_id, with_for_update=True)
    if dataset is None:
        raise ValueError(f"no such dataset: {dataset_id}")

    byte_size = len(content.encode("utf-8"))
    row = Document(
        dataset_id=dataset_id,
        ordinal=dataset.document_count,
        cache_key=cache_key,
        content=content,
        meta=meta,
        byte_size=byte_size,
    )
    db.add(row)
    dataset.document_count += 1
    dataset.byte_size += byte_size
    dataset.updated_at = _now()
    db.flush()
    return row


def gym_dataset(db: Session, name: str, description: str, steps: list) -> Dataset:
    """The dataset a gym's demonstrations land in, created on first write.

    Named by the caller rather than reserved as a constant the way the search
    cache is: there is one cache, and there are as many demonstration sets as
    there are task types and difficulties worth training on separately. A second
    request under the same name **appends**, which is what makes a set growable
    without a re-pull — and what makes `updated_at` and `document_count`
    load-bearing, since the engine's corpus key reads both.

    Immediately `ready`. Every row is complete when it is written; there is no
    batch to wait out, so there is no interval where calling this half-written
    would be true.
    """
    row = dataset_by_name(db, name)
    if row is not None:
        return row
    try:
        with db.begin_nested():
            return create_dataset(
                db,
                name=name,
                kind=DatasetKind.sft,
                origin=DatasetOrigin.gym,
                # A demonstration is a fact about a task, not about whoever
                # asked for it — the same reasoning the search cache uses.
                owner_id=None,
                description=description,
                steps=steps,
                status=DatasetStatus.ready,
            )
    except IntegrityError:
        # Lost a race with another writer creating the same row; its commit is
        # the one that landed.
        row = dataset_by_name(db, name)
        if row is None:
            raise
        return row


def add_conversations(db: Session, dataset_id: uuid.UUID, rows: list[dict]) -> int:
    """Append conversations outside the batch `Run` path, numbering from the end.

    The batched sibling of `add_document`, and it holds the same lock for the
    same reason: `ordinal` is assigned from `document_count`, so two concurrent
    writers without it would hand out the same ordinals and silently interleave
    two demonstration sets. `finish_dataset`'s recompute is not available here —
    there is no run to end — so the counters are bumped in place.
    """
    if not rows:
        return 0

    dataset = db.get(Dataset, dataset_id, with_for_update=True)
    if dataset is None:
        raise ValueError(f"no such dataset: {dataset_id}")

    start = dataset.document_count
    payload = []
    written = 0
    for offset, row in enumerate(rows):
        messages = row["messages"]
        byte_size = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
        payload.append(
            {
                "id": uuid.uuid4(),
                "dataset_id": dataset_id,
                "ordinal": start + offset,
                "content": None,
                "messages": messages,
                "meta": row.get("meta"),
                "byte_size": byte_size,
            }
        )
        written += byte_size

    db.execute(Document.__table__.insert(), payload)
    dataset.document_count += len(payload)
    dataset.byte_size += written
    dataset.updated_at = _now()
    db.flush()
    return len(payload)


def list_runs(db: Session, limit: int) -> list[Run]:
    return list(db.scalars(select(Run).order_by(Run.created_at.desc()).limit(limit)))


def run(db: Session, run_id: uuid.UUID) -> Run | None:
    return db.get(Run, run_id)


def create_run(
    db: Session,
    *,
    kind: RunKind,
    label: str,
    config: dict,
    owner_id: uuid.UUID | None,
    ref_id: uuid.UUID | None = None,
) -> Run:
    row = Run(kind=kind, label=label, config=config, owner_id=owner_id, ref_id=ref_id)
    db.add(row)
    db.flush()
    return row


def claim_next(db: Session) -> Run | None:
    """Take the oldest queued run and mark it running, in one transaction.

    Claiming and starting are the same act on purpose: a row read as queued and
    marked running in a later transaction can be claimed twice, and one runner
    today is not a reason to write a queue that breaks when there are two.
    """
    row = db.scalar(
        select(Run)
        .where(Run.status == RunStatus.queued)
        .order_by(Run.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    if row is None:
        return None
    row.status = RunStatus.running
    row.started_at = _now()
    return row


def report_progress(
    db: Session, run_id: uuid.UUID, progress: float | None, stage: str | None
) -> None:
    """Update the coarse bar, leaving alone whatever the frame did not measure.

    A frame that carries a stage but no fraction must not reset the bar to zero.
    """
    row = db.get(Run, run_id)
    if row is None:
        return
    if progress is not None:
        row.progress = progress
    if stage is not None:
        row.stage = stage


def finish_run(
    db: Session,
    run_id: uuid.UUID,
    status: RunStatus,
    *,
    result: dict | None = None,
    error: str | None = None,
) -> None:
    row = db.get(Run, run_id)
    if row is not None:
        row.status = status
        row.result = result
        row.error = error
        row.finished_at = _now()


def request_cancel(db: Session, run_id: uuid.UUID) -> bool:
    """Ask a run to stop. False when it is already over and there is nothing to
    ask — the caller turns that into a 409 rather than pretending it worked."""
    row = db.get(Run, run_id)
    if row is None or row.status.is_terminal:
        return False
    row.status = RunStatus.cancelling
    return True


def fail_orphaned(db: Session) -> int:
    """Close out runs still marked live from a previous process.

    Their worker died with it. Queued rows are deliberately untouched — the
    runner drains them, which is the point of the queue surviving a restart.

    Datasets left mid-pull go with them: rows exist for them, but not all of the
    rows, and a corpus that is silently short is worse than one marked broken.
    """
    stranded = list(
        db.scalars(select(Run).where(Run.status.in_((RunStatus.running, RunStatus.cancelling))))
    )
    for row in stranded:
        row.status = RunStatus.failed
        row.error = "the data engine restarted while this run was in flight"
        row.finished_at = _now()

    for row in db.scalars(select(Dataset).where(Dataset.status == DatasetStatus.pulling)):
        row.status = DatasetStatus.failed

    return len(stranded)
