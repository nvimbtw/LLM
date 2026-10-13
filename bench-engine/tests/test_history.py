"""History contracts moved with their database owner."""
import uuid
import pytest
import store

@pytest.fixture
def scored(db):
    with store.session() as session:
        first,second = uuid.uuid4(),uuid.uuid4()
        a = store.Benchmark(model_id=first,model_name='first',suite='language',score=30.0,status='done',report={'detail':'kept off the list'})
        session.add_all([a,store.Benchmark(model_id=second,model_name='second',suite='language',score=40.0,status='done'),
                         store.Benchmark(model_id=second,model_name='second',suite='markup',score=50.0,status='done')])
        session.flush()
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
    assert body["chat_files"] == 0
    assert body["corpus_dir"].endswith("corpus")


async def test_historical_suites_cannot_be_queued(client):
    response = await client.post(f'/execute/{uuid.uuid4()}',json={'model_name':str(uuid.uuid4()),'suite':'markup'})
    assert response.status_code == 400
