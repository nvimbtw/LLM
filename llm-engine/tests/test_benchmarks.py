"""Scores across models, and what the caller is told when a suite is missing."""

import uuid

import pytest

import store
from api import auth
from store import queries as q
from store.schema import BenchmarkStatus, Suite


@pytest.fixture
def client(engine_client):
    import main

    main.app.dependency_overrides[auth.require_service] = lambda: auth.SYSTEM
    yield engine_client
    main.app.dependency_overrides.clear()


@pytest.fixture
def scored():
    """Two models, each with a language score and one with a markup score."""
    with store.session() as db:
        first = q.upsert_model(db, model_id=uuid.uuid4(), name="first")
        second = q.upsert_model(db, model_id=uuid.uuid4(), name="second")
        a = q.open_benchmark(
            db,
            model=first,
            suite=Suite.language,
            score=30.0,
            status=BenchmarkStatus.done,
            report={"detail": "kept off the list"},
        )
        q.open_benchmark(db, model=second, suite=Suite.language, score=40.0)
        q.open_benchmark(db, model=second, suite=Suite.markup, score=50.0)
        return a.id


async def test_the_history_is_of_one_suite(client, scored):
    """Suites share a range and nothing else, so they are never listed together."""
    language = (await client.get("/benchmarks?suite=language")).json()["benchmarks"]
    markup = (await client.get("/benchmarks?suite=markup")).json()["benchmarks"]

    assert sorted(b["score"] for b in language) == [30.0, 40.0]
    assert [b["score"] for b in markup] == [50.0]


async def test_the_list_leaves_the_report_out(client, scored):
    """A report is a page of JSON per row; the list is an index."""
    listed = (await client.get("/benchmarks")).json()["benchmarks"]
    assert all("report" not in b for b in listed)


async def test_one_run_carries_its_whole_report(client, scored):
    body = (await client.get(f"/benchmarks/{scored}")).json()
    assert body["report"] == {"detail": "kept off the list"}


async def test_an_absent_run_404s(client):
    absent = uuid.uuid4()
    assert (await client.get(f"/benchmarks/{absent}")).status_code == 404
    assert (await client.delete(f"/benchmarks/{absent}")).status_code == 404


async def test_deleting_a_run_removes_only_that_run(client, scored):
    assert (await client.delete(f"/benchmarks/{scored}")).json()["deleted"] is True

    remaining = (await client.get("/benchmarks?suite=language")).json()["benchmarks"]
    assert [b["score"] for b in remaining] == [40.0]


async def test_the_suite_reports_what_is_on_disk(client):
    """So an absent score reads as "no corpus here" rather than as a bug."""
    body = (await client.get("/benchmarks/suite")).json()
    assert body["corpus_files"] == 1
    assert body["chat_files"] == 1
    assert body["corpus_dir"].endswith("eval/corpus")
