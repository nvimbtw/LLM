"""The cache a web-search tool call checks before paying for a live search."""

from store import queries as q


async def test_a_miss_is_an_empty_hit_not_a_404(client):
    response = await client.post("/search-cache/lookup", json={"key": "tallest mountain europe"})

    assert response.status_code == 200
    assert response.json() == {"content": None, "created_at": None}


async def test_a_stored_result_is_found_by_its_key(client):
    stored = await client.post(
        "/search-cache", json={"key": "tallest mountain europe", "content": "Mount Elbrus"}
    )
    assert stored.status_code == 201

    found = await client.post("/search-cache/lookup", json={"key": "tallest mountain europe"})
    body = found.json()

    assert body["content"] == "Mount Elbrus"
    assert body["created_at"] is not None


async def test_a_different_key_is_a_separate_miss(client):
    await client.post("/search-cache", json={"key": "a", "content": "answer to a"})

    found = await client.post("/search-cache/lookup", json={"key": "b"})

    assert found.json()["content"] is None


async def test_a_re_fetch_appends_rather_than_overwrites(client, store_session):
    import store

    await client.post("/search-cache", json={"key": "k", "content": "first answer"})
    await client.post("/search-cache", json={"key": "k", "content": "second answer"})

    latest = await client.post("/search-cache/lookup", json={"key": "k"})
    assert latest.json()["content"] == "second answer"

    with store.session() as db:
        dataset = q.dataset_by_name(db, q.CACHE_DATASET_NAME)
        rows = q.documents(db, dataset.id, offset=0, limit=10)
        assert [r.content for r in rows] == ["first answer", "second answer"]


async def test_the_cache_dataset_is_unowned_and_ready_immediately(store_session):
    import store
    from store.schema import DatasetOrigin, DatasetStatus

    with store.session() as db:
        dataset = q.cache_dataset(db)

    assert dataset.owner_id is None
    assert dataset.origin is DatasetOrigin.search
    assert dataset.status is DatasetStatus.ready


async def test_cache_dataset_is_idempotent_across_calls(store_session):
    import store

    with store.session() as db:
        first = q.cache_dataset(db)
        second = q.cache_dataset(db)

    assert first.id == second.id
