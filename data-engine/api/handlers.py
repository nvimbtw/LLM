"""What the queue actually runs.

Two kinds of work, one shape: read documents from somewhere, write them into a
new dataset in batches, report how far along it is, and stop when asked. The
difference between a pull and a process is only where the documents come from,
so the writing half is shared.

A handler runs on a worker thread, so it opens its own sessions and never
touches the event loop.
"""

import threading
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import hub
import processors
import store
from store import queries as q
from store.schema import DatasetKind, DatasetStatus, RunKind

#: Documents per transaction. Large enough that a big pull is not dominated by
#: round trips, small enough that a cancel lands promptly and a failure loses
#: little.
WRITE_BATCH = 200

#: Documents read per page when processing. Independent of `WRITE_BATCH`: the
#: reader and the writer are separate transactions on purpose.
READ_PAGE = 500

#: How many rows a pull will look at before deciding the column it was told to
#: read does not exist. Some sources open with a run of empty rows.
COLUMN_PATIENCE = 200

#: A pull must name a bound. This is what it gets when the caller's is missing or
#: nonsense — generous enough to be useful, small enough to be survivable.
DEFAULT_ROW_CAP = 10_000

#: The ceiling a caller cannot raise. Lifting it is a deliberate edit here, not a
#: number typed into a form at three in the morning.
MAX_ROW_CAP = 5_000_000

Report = Callable[[Any], None]


def _cap(config: dict) -> int:
    requested = config.get("limit_rows")
    if not isinstance(requested, int) or requested <= 0:
        return DEFAULT_ROW_CAP
    return min(requested, MAX_ROW_CAP)


def _write(
    documents: Iterator[dict],
    dataset_id: uuid.UUID,
    cap: int,
    report: Report,
    cancel: threading.Event,
    stage: str,
) -> tuple[int, bool]:
    """Drain `documents` into a dataset, in batches, until the cap or a cancel.

    Returns how many landed and whether it was cancelled. The caller decides what
    that means — a cancelled pull keeps what it got rather than throwing the work
    away, and says so.
    """
    written = 0
    batch: list[dict] = []

    for document in documents:
        if cancel.is_set():
            break
        batch.append(document)

        # The cap bounds the *intake*, not the batch that happens to cross it.
        # Checking only at batch boundaries silently rounds every request up to
        # WRITE_BATCH — asking for 40 documents and being given 200.
        full = written + len(batch) >= cap
        if len(batch) < WRITE_BATCH and not full:
            continue

        written += _commit(dataset_id, written, batch[: cap - written])
        batch = []
        report({"stage": stage, "documents": written, "fraction": min(written / cap, 1.0)})
        if written >= cap:
            break

    if batch and not cancel.is_set():
        written += _commit(dataset_id, written, batch[: max(cap - written, 0)])

    report({"stage": stage, "documents": written, "fraction": min(written / cap, 1.0)})
    return written, cancel.is_set()


def _condemn(dataset_id: uuid.UUID) -> None:
    """Mark a half-written dataset failed, on its own session.

    Its own because the session that raised is gone and its transaction with it,
    and a best effort because this runs while an exception is in flight: failing
    to record the failure must not replace the error that caused it.
    """
    try:
        with store.session() as db:
            q.finish_dataset(db, dataset_id, DatasetStatus.failed)
    except Exception as e:
        print(f"handlers: could not condemn {dataset_id}: {type(e).__name__}: {e}")


def _commit(dataset_id: uuid.UUID, start: int, batch: list[dict]) -> int:
    if not batch:
        return 0
    with store.session() as db:
        return q.add_documents(db, dataset_id, start, batch)


def pull(config: dict, report: Report, cancel: threading.Event) -> dict:
    """Fetch rows from the Hub into the dataset the route already created.

    The row exists before the run starts so the page can show it as `pulling`;
    this fills it and then publishes it.
    """
    dataset_id = uuid.UUID(str(config["dataset_id"]))
    repo = str(config["repo"])
    kind = DatasetKind(config.get("kind", DatasetKind.pretrain.value))
    cap = _cap(config)

    report({"stage": "listing shards", "fraction": 0.0})
    shards = hub.shards_under(repo, config.get("prefix") or None)

    named = config.get("column") or None
    wanted = [named] if named else None
    rows = hub.iter_rows(repo, shards, wanted)

    documents = _documents_from(rows, kind, named)
    try:
        written, cancelled = _write(documents, dataset_id, cap, report, cancel, "pulling")
    except Exception:
        _condemn(dataset_id)
        raise

    with store.session() as db:
        q.finish_dataset(
            db, dataset_id, DatasetStatus.ready if written else DatasetStatus.failed
        )

    return {
        "dataset_id": str(dataset_id),
        "documents": written,
        "cancelled": cancelled,
        "shards": len(shards),
    }


def _documents_from(rows: Iterator[dict], kind: DatasetKind, named: str | None) -> Iterator[dict]:
    """Shape Hub rows into documents, resolving the content column from the first
    row that has one.

    Resolved once and then held: a shard whose first row is empty should not
    change which column the rest of the pull reads. Failing to resolve one at all
    is fatal and says what the row actually offered — a pull that quietly writes
    nothing looks identical to a dataset that is genuinely empty.
    """
    candidates = hub.TEXT_COLUMNS if kind == DatasetKind.pretrain else hub.MESSAGE_COLUMNS
    column: str | None = None
    last: dict | None = None
    looked_at = 0

    for row in rows:
        if column is None:
            last = row
            column = hub.pick_column(row, named, candidates)
            if column is None:
                # Patience, because some sources open with a run of rows whose
                # only populated columns are metadata.
                looked_at += 1
                if looked_at >= COLUMN_PATIENCE:
                    raise _no_column(kind, named, row)
                continue
        document = hub.as_document(row, kind, column)
        if document is not None:
            yield document

    # Running out of rows without ever resolving one is the same failure, and it
    # is the one a source smaller than COLUMN_PATIENCE hits. Silence here looks
    # exactly like a dataset that is genuinely empty.
    if column is None and last is not None:
        raise _no_column(kind, named, last)


def _no_column(kind: DatasetKind, named: str | None, row: dict) -> Exception:
    asked = f" (asked for {named!r})" if named else ""
    return hub.HubError(f"no {kind.value} column in {sorted(row)}{asked}")


def process(config: dict, report: Report, cancel: threading.Event) -> dict:
    """Run a pipeline over one dataset into another.

    Reads the source in order and streams it, so cleaning a corpus costs the same
    memory whether it has a thousand documents or a million.
    """
    source_id = uuid.UUID(str(config["source_id"]))
    target_id = uuid.UUID(str(config["target_id"]))
    steps = processors.resolve(list(config.get("steps") or []), processors.REGISTRY)

    with store.session() as db:
        source = q.dataset(db, source_id)
        if source is None:
            raise LookupError(f"dataset {source_id} is gone")
        total = source.document_count

    seen = 0
    state: processors.State = {}

    def cleaned() -> Iterator[dict]:
        """The source, cleaned, read a page at a time.

        Paged rather than streamed because the writer commits while this is
        iterating: holding a read transaction open across those writes deadlocks
        on one connection, which is exactly what the tests run on.
        """
        nonlocal seen
        offset = 0
        while True:
            with store.session() as db:
                page = [
                    _as_document(row)
                    for row in q.documents(db, source_id, offset=offset, limit=READ_PAGE)
                ]
            if not page:
                return

            offset += len(page)
            for document in page:
                if cancel.is_set():
                    return
                seen += 1
                outcome = processors.run(steps, document, state)
                if outcome.after is not None:
                    yield _sized(outcome.after)

    # The cap is the source's own size: a pipeline cannot invent documents, so
    # this bounds nothing in practice and keeps one write path for both kinds.
    try:
        written, cancelled = _write(
            cleaned(), target_id, max(total, 1), report, cancel, "processing"
        )
    except Exception:
        # A dataset left in `pulling` is a dataset that never stops: the row is
        # what the page reads, and nothing else ever comes back to close it. The
        # run is marked failed by the runner and carries the traceback; this is
        # only so the dataset says so too, with a count of what actually landed
        # rather than the zero it was created with.
        _condemn(target_id)
        raise

    with store.session() as db:
        q.finish_dataset(
            db, target_id, DatasetStatus.failed if cancelled and not written else DatasetStatus.ready
        )

    return {
        "dataset_id": str(target_id),
        "documents": written,
        "read": seen,
        "dropped": max(seen - written, 0),
        "cancelled": cancelled,
    }


def _as_document(row: store.Document) -> dict:
    return {"content": row.content, "messages": row.messages, "meta": row.meta}


def _sized(document: dict) -> dict:
    """A document with its stored size recomputed, since a processor changed it."""
    if document.get("content") is not None:
        size = len((document["content"] or "").encode("utf-8"))
    else:
        size = sum(
            len(str(m.get("content") or "").encode("utf-8"))
            for m in (document.get("messages") or [])
        )
    return {**document, "byte_size": size}


HANDLERS = {RunKind.pull: pull, RunKind.process: process}
