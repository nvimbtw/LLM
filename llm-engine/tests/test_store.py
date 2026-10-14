"""The LLM engine's database: what it is the authority on, and what it refuses."""

import uuid

import pytest

from store import queries as q
from store.schema import Run, RunKind, RunStatus

ALICE = uuid.uuid4()
BOB = uuid.uuid4()


@pytest.fixture
def db(store_session):
    with store_session() as session:
        yield session


@pytest.fixture
def model(db):
    return q.upsert_model(db, model_id=uuid.uuid4(), name="tiny-1", context=128)


# ── models ──────────────────────────────────────────────────────────────────


def test_a_model_is_private_until_published(db, model):
    assert model.published is False
    assert q.list_models(db, include_private=False) == []
    assert q.list_models(db, include_private=True) == [model]


def test_retraining_the_same_id_updates_rather_than_collides(db, model):
    """The id is the weights directory, so a retrain writes the same row —
    inserting instead would violate the unique name."""
    again = q.upsert_model(db, model_id=model.id, name="tiny-1", context=256)

    assert again.id == model.id
    assert again.context == 256
    assert len(q.list_models(db, include_private=True)) == 1


def test_a_second_model_may_be_given_a_name_that_is_taken(db, model):
    """Ids are minted per queued run, so queueing the same fine-tune twice is two
    models with one label — the ordinary way to retry a cancelled run.

    Registration happens after the weights are on disk, so a clash there used to
    raise `UniqueViolation` and lose the row for a run that had finished
    training. The name gives way; the model does not."""
    second = q.upsert_model(db, model_id=uuid.uuid4(), name="tiny-1", context=64)
    third = q.upsert_model(db, model_id=uuid.uuid4(), name="tiny-1", context=64)

    assert second.name == "tiny-1 (2)"
    assert third.name == "tiny-1 (3)"
    assert q.find_model_by_name(db, "tiny-1").id == model.id
    assert len(q.list_models(db, include_private=True)) == 3


def test_rewriting_a_model_keeps_the_name_it_was_given(db, model):
    """Only *another* id counts as a clash. A row written twice — a run
    re-reporting, a field corrected — must not climb a suffix each time."""
    q.upsert_model(db, model_id=model.id, name="tiny-1", context=64)
    q.upsert_model(db, model_id=model.id, name="tiny-1", context=32)

    assert model.name == "tiny-1"


def test_deleting_a_model_reports_the_name_it_had(db, model):
    assert q.delete_model(db, model.id) == "tiny-1"
    assert q.delete_model(db, model.id) is None


# ── the queue ───────────────────────────────────────────────────────────────


def enqueue(db, label: str):
    return q.enqueue(db, kind=RunKind.train, label=label, config={})


def test_runs_are_claimed_in_rank_order(db):
    first, second = enqueue(db, "first"), enqueue(db, "second")

    assert q.claim_next(db) is first
    assert q.claim_next(db) is second
    assert q.claim_next(db) is None


def test_claiming_marks_it_running_and_stamps_the_start(db):
    enqueue(db, "one")
    run = q.claim_next(db)

    assert run.status is RunStatus.running
    assert run.started_at is not None




def test_a_waiting_run_is_never_claimed(db):
    """It consumes a model that does not exist yet; claiming it would fail."""
    q.enqueue(db, kind=RunKind.train, label="follow-up", config={}, status=RunStatus.waiting)
    assert q.claim_next(db) is None


def test_reorder_moves_one_run_between_its_new_neighbours(db):
    a, b, c = enqueue(db, "a"), enqueue(db, "b"), enqueue(db, "c")

    assert q.reorder(db, c.id, after=a.id) is True
    assert [r.label for r in q.active_runs(db)] == ["a", "c", "b"]
    assert a.rank < c.rank < b.rank


def test_reorder_to_the_front(db):
    a, b = enqueue(db, "a"), enqueue(db, "b")

    assert q.reorder(db, b.id, after=None) is True
    assert [r.label for r in q.active_runs(db)] == ["b", "a"]


def test_reorder_renumbers_when_doubles_run_out_of_midpoints(db):
    """Bisecting the same gap forever eventually cannot represent a midpoint."""
    a, b, c = enqueue(db, "a"), enqueue(db, "b"), enqueue(db, "c")
    b.rank = a.rank + q.MIN_RANK_GAP / 4
    c.rank = a.rank + q.MIN_RANK_GAP / 2
    db.flush()

    assert q.reorder(db, c.id, after=a.id) is True
    ranks = sorted(r.rank for r in q.active_runs(db))
    assert all(hi - lo >= q.MIN_RANK_GAP for lo, hi in zip(ranks, ranks[1:]))


def test_a_running_run_cannot_be_reordered(db):
    enqueue(db, "a")
    running = q.claim_next(db)
    assert q.reorder(db, running.id, after=None) is False


def test_finishing_records_what_it_produced(db):
    enqueue(db, "a")
    run = q.claim_next(db)
    produced = uuid.uuid4()

    q.finish_run(db, run.id, RunStatus.done, ref_id=produced)
    assert run.status is RunStatus.done
    assert run.ref_id == produced
    assert run.finished_at is not None


def test_a_failed_reader_does_not_roll_back_a_worker_write(store_session):
    """Overlapping request and worker transactions retain independent outcomes."""
    with store_session() as worker:
        run = q.enqueue(worker, kind=RunKind.train, label="worker", config={})
        run_id = run.id
        with pytest.raises(RuntimeError, match="reader failed"):
            with store_session() as reader:
                reader.get(Run, run_id)
                raise RuntimeError("reader failed")

    with store_session() as reader:
        assert reader.get(Run, run_id) is not None


def test_orphans_are_failed_at_startup_but_the_queue_survives(db):
    enqueue(db, "was-running")
    q.claim_next(db)
    still_queued = enqueue(db, "still-queued")

    assert q.fail_orphaned(db) == 1
    assert still_queued.status is RunStatus.queued
    assert q.claim_next(db) is still_queued


def test_history_holds_only_terminal_runs(db):
    enqueue(db, "finished")
    done = q.claim_next(db)
    q.finish_run(db, done.id, RunStatus.done)
    enqueue(db, "still-queued")

    assert [r.label for r in q.run_history(db)] == ["finished"]


def test_a_dependent_run_is_found_by_what_it_waits_on(db):
    parent = enqueue(db, "pretrain")
    child = q.enqueue(
        db,
        kind=RunKind.train,
        label="fine-tune",
        config={},
        status=RunStatus.waiting,
        depends_on=parent.id,
    )

    assert q.waiting_on(db, parent.id) == [child]
