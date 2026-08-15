"""The engine's database: what it is the authority on, and what it refuses."""

import uuid

import pytest

from store import queries as q
from store.schema import Feedback, Role, RunKind, RunStatus, Suite

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


# ── chats: the ownership boundary ───────────────────────────────────────────


def test_a_chat_is_only_reachable_by_its_owner(db):
    chat = q.create_chat(db, ALICE, "mine")

    assert q.owned_chat(db, chat.id, ALICE) is chat
    assert q.owned_chat(db, chat.id, BOB) is None


def test_listing_is_scoped_to_one_owner(db):
    q.create_chat(db, ALICE, "a")
    q.create_chat(db, BOB, "b")

    assert [c.title for c, _ in q.list_chats(db, ALICE)] == ["a"]


def test_a_chats_tokens_are_summed_from_its_messages(db, model):
    chat = q.create_chat(db, ALICE, "counted")
    q.add_message(db, role=Role.user, chat_id=chat.id, content="hi", tokens=3)
    q.add_message(db, role=Role.assistant, chat_id=chat.id, content="hello", tokens=4, model_id=model.id)

    assert q.list_chats(db, ALICE) == [(chat, 7)]


def test_an_empty_chat_reports_zero_rather_than_vanishing(db):
    chat = q.create_chat(db, ALICE, "empty")
    assert q.list_chats(db, ALICE) == [(chat, 0)]


def test_renaming_and_deleting_refuse_a_stranger(db):
    chat = q.create_chat(db, ALICE, "mine")

    assert q.rename_chat(db, chat.id, BOB, "hijacked") is None
    assert q.delete_chat(db, chat.id, BOB) is False
    assert q.owned_chat(db, chat.id, ALICE).title == "mine"


def test_deleting_a_chat_takes_its_messages(db):
    chat = q.create_chat(db, ALICE, "doomed")
    q.add_message(db, role=Role.user, chat_id=chat.id, content="hi")
    db.flush()

    assert q.delete_chat(db, chat.id, ALICE) is True
    db.flush()
    assert q.messages_of(db, chat.id) == []


def test_deleting_an_owner_is_idempotent(db):
    """The backend deletes the account after this returns, so a half-failed
    deletion is fixed by retrying rather than by reconciling."""
    q.create_chat(db, ALICE, "a")
    q.create_chat(db, ALICE, "b")

    assert q.delete_chats_of(db, ALICE) == 2
    assert q.delete_chats_of(db, ALICE) == 0


def test_distinct_owners_backs_the_reconciliation_sweep(db):
    q.create_chat(db, ALICE, "a")
    q.create_chat(db, ALICE, "b")
    q.create_chat(db, BOB, "c")

    assert set(q.distinct_owners(db)) == {ALICE, BOB}


# ── messages ────────────────────────────────────────────────────────────────


def test_authorship_is_encoded_by_which_id_is_set(db, model):
    chat = q.create_chat(db, ALICE, "c")
    user = q.add_message(db, role=Role.user, chat_id=chat.id, content="q")
    reply = q.add_message(db, role=Role.assistant, chat_id=chat.id, content="a", model_id=model.id)

    assert user.role == "user"
    assert reply.role == "assistant"


def test_feedback_starts_unrated_and_is_scoped_to_the_owner(db, model):
    chat = q.create_chat(db, ALICE, "c")
    reply = q.add_message(db, role=Role.assistant, chat_id=chat.id, content="a", model_id=model.id)

    assert reply.feedback is None
    assert q.set_feedback(db, reply.id, BOB, Feedback.up) is False
    assert q.set_feedback(db, reply.id, ALICE, Feedback.up) is True
    assert reply.feedback is Feedback.up


def test_feedback_can_be_cleared(db, model):
    chat = q.create_chat(db, ALICE, "c")
    reply = q.add_message(db, role=Role.assistant, chat_id=chat.id, content="a", model_id=model.id)
    q.set_feedback(db, reply.id, ALICE, Feedback.down)

    assert q.set_feedback(db, reply.id, ALICE, None) is True
    assert reply.feedback is None


# ── benchmarks ──────────────────────────────────────────────────────────────


def test_a_benchmark_records_the_name_the_model_had(db, model):
    run = q.open_benchmark(db, model=model, suite=Suite.language)
    model.name = "renamed"
    db.flush()

    assert run.model_name == "tiny-1"


def test_the_latest_score_is_per_suite_and_per_version(db, model):
    q.open_benchmark(db, model=model, suite=Suite.language, suite_version=1, score=40.0)
    newer = q.open_benchmark(db, model=model, suite=Suite.language, suite_version=2, score=55.0)
    q.open_benchmark(db, model=model, suite=Suite.markup, suite_version=2, score=99.0)

    assert q.latest_benchmark(db, model.id, Suite.language, 2) is newer
    assert q.latest_benchmark(db, model.id, Suite.assistant, 2) is None


def test_a_run_without_a_score_is_not_the_latest(db, model):
    """A failed or in-flight run must not be read as a measurement."""
    q.open_benchmark(db, model=model, suite=Suite.language, suite_version=1, score=40.0)
    q.open_benchmark(db, model=model, suite=Suite.language, suite_version=1)

    assert q.latest_benchmark(db, model.id, Suite.language, 1).score == 40.0


def test_deleting_a_model_takes_its_benchmarks(db, model):
    q.open_benchmark(db, model=model, suite=Suite.language)
    q.delete_model(db, model.id)
    db.flush()

    assert q.benchmarks_of(db, model.id) == []


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


def test_a_paused_queue_hands_out_nothing(db):
    enqueue(db, "one")
    q.set_paused(db, True)
    assert q.claim_next(db) is None

    q.set_paused(db, False)
    assert q.claim_next(db) is not None


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
