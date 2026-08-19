"""Reads and writes, one function per question.

Everything that scopes by owner takes it as an argument rather than reading it
from anywhere — this service cannot resolve an owner, so forgetting to pass one
has to be a type error rather than a silent full-table read.

Nothing here returns a whole dataset. A corpus-sized table is read by page or
streamed; the functions that touch `documents` all take a bound.
"""

import uuid
from collections.abc import Iterator
from datetime import datetime, timezone

from sqlalchemy import delete, func, select
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
