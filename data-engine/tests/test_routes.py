"""The HTTP surface, over a real app and a real (if small) database."""

import asyncio
import uuid


async def test_health_answers_without_a_token(client):
    response = await client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_a_dataset_lists_and_pages_its_documents(client, acting, pretrain_dataset):
    listed = await client.get("/datasets", headers=acting)
    assert [d["name"] for d in listed.json()["datasets"]] == ["fixture-pretrain"]

    page = await client.get(
        f"/datasets/{pretrain_dataset}/documents?offset=1&limit=2", headers=acting
    )
    body = page.json()

    assert body["total"] == 4
    assert [d["ordinal"] for d in body["documents"]] == [1, 2]


async def test_an_unknown_dataset_is_a_404_not_a_500(client, acting):
    response = await client.get(f"/datasets/{uuid.uuid4()}", headers=acting)

    assert response.status_code == 404


async def test_the_processor_catalogue_is_filtered_by_kind(client, acting):
    sft = await client.get("/processors?kind=sft", headers=acting)
    names = {p["name"] for p in sft.json()["processors"]}

    assert "normalise-roles" in names
    assert "strip-html" not in names


async def test_preview_reports_both_sides_and_what_was_dropped(
    client, acting, pretrain_dataset
):
    response = await client.post(
        "/preview",
        headers=acting,
        json={
            "dataset_id": str(pretrain_dataset),
            "limit": 10,
            "steps": [
                {"processor": "strip-html"},
                {"processor": "collapse-whitespace"},
                {"processor": "min-length", "params": {"characters": 10}},
            ],
        },
    )
    body = response.json()

    assert body["sampled"] == 4
    # "short" is below the bound; "Hello world" is eleven characters and survives.
    assert body["dropped"] == 1
    dropped = next(r for r in body["results"] if r["dropped"])
    assert dropped["dropped_by"] == "min-length"

    kept = next(r for r in body["results"] if not r["dropped"])
    assert "<p>" in kept["before"]["content"]
    assert "<p>" not in kept["after"]["content"]


async def test_preview_is_stable_across_calls(client, acting, pretrain_dataset):
    """The user tunes a parameter and compares runs, so the sample must not move
    underneath them."""
    body = {
        "dataset_id": str(pretrain_dataset),
        "limit": 2,
        "steps": [{"processor": "collapse-whitespace"}],
    }

    first = await client.post("/preview", headers=acting, json=body)
    second = await client.post("/preview", headers=acting, json=body)

    assert [r["ordinal"] for r in first.json()["results"]] == [0, 1]
    assert first.json()["results"] == second.json()["results"]


async def test_preview_refuses_an_unknown_processor(client, acting, pretrain_dataset):
    response = await client.post(
        "/preview",
        headers=acting,
        json={"dataset_id": str(pretrain_dataset), "steps": [{"processor": "invent-data"}]},
    )

    assert response.status_code == 422


async def test_processing_writes_a_derived_dataset_and_leaves_the_source_alone(
    client, acting, pretrain_dataset
):
    queued = await client.post(
        f"/datasets/{pretrain_dataset}/process",
        headers=acting,
        json={
            "name": "fixture-cleaned",
            "steps": [
                {"processor": "strip-html"},
                {"processor": "collapse-whitespace"},
                {"processor": "min-length", "params": {"characters": 10}},
                {"processor": "dedup-exact"},
            ],
        },
    )
    assert queued.status_code == 202
    derived_id = queued.json()["dataset_id"]

    await _settled(client, acting, queued.json()["run_id"])

    derived = (await client.get(f"/datasets/{derived_id}", headers=acting)).json()
    source = (await client.get(f"/datasets/{pretrain_dataset}", headers=acting)).json()

    # Four documents in: one too short, one an exact duplicate of another.
    assert derived["document_count"] == 2
    assert derived["origin"] == "derived"
    assert derived["parent_id"] == str(pretrain_dataset)
    assert [s["processor"] for s in derived["steps"]] == [
        "strip-html", "collapse-whitespace", "min-length", "dedup-exact"
    ]
    # Provenance records the parameters used, not the ones typed.
    assert derived["steps"][2]["params"] == {"characters": 10}
    assert source["document_count"] == 4


async def test_processing_refuses_a_name_already_taken(client, acting, pretrain_dataset):
    response = await client.post(
        f"/datasets/{pretrain_dataset}/process",
        headers=acting,
        json={"name": "fixture-pretrain", "steps": [{"processor": "collapse-whitespace"}]},
    )

    assert response.status_code == 409


async def test_processing_refuses_a_processor_for_the_wrong_kind(
    client, acting, pretrain_dataset
):
    response = await client.post(
        f"/datasets/{pretrain_dataset}/process",
        headers=acting,
        json={"name": "fixture-wrong", "steps": [{"processor": "normalise-roles"}]},
    )

    assert response.status_code == 422


async def test_export_streams_the_documents_as_json_lines(
    client, acting, pretrain_dataset
):
    response = await client.get(f"/datasets/{pretrain_dataset}/export", headers=acting)
    lines = [line for line in response.text.splitlines() if line]

    assert response.headers["x-dataset-kind"] == "pretrain"
    assert len(lines) == 4
    assert '"text"' in lines[0]


async def test_deleting_a_dataset_takes_its_documents_with_it(
    client, acting, pretrain_dataset
):
    removed = await client.delete(f"/datasets/{pretrain_dataset}", headers=acting)
    assert removed.status_code == 204

    import store
    from store import queries as q

    with store.session() as db:
        assert q.documents(db, pretrain_dataset, offset=0, limit=10) == []


async def test_a_mutating_route_refuses_the_backend_acting_as_itself(client):
    """No `act` claim means the backend called for itself, and it owns nothing."""
    response = await client.post(
        "/datasets/pull", json={"repo": "a/b", "name": "anything"}
    )

    assert response.status_code == 403


async def _settled(client, acting, run_id: str, tries: int = 200) -> dict:
    """Poll until the run reaches a terminal state, then return it."""
    for _ in range(tries):
        run = (await client.get(f"/runs/{run_id}", headers=acting)).json()
        if run["status"] in ("done", "failed", "cancelled"):
            return run
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} never settled")
