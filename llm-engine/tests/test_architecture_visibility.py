"""Signed authorization filters private validation before pagination, serialization and streams."""

from datetime import UTC, datetime, timedelta
import json
from uuid import uuid4

import jwt
import pytest

import store
from api import auth
from api.runs import Event, REGISTRY
from store import queries as q
from store.schema import RunKind, RunStatus

SECRET = "visibility-test-secret"
SOURCE = "private_python_source_and_captured_graph"


def headers(scope: object = None, *, key: str = SECRET) -> dict[str, str]:
    """Mint the exact Rust service-hop shape, including actor and optional visibility scope."""
    claims = {
        "iss": "backend", "aud": "llm-engine", "act": str(uuid4()),
        "exp": datetime.now(UTC) + timedelta(minutes=1),
    }
    if scope is not None:
        claims["scope"] = scope
    return {"Authorization": "Bearer " + jwt.encode(claims, key, algorithm="HS256")}


@pytest.fixture
async def client(engine_client, monkeypatch):
    """Keep the real authorization dependencies while pausing numerical queue execution."""
    monkeypatch.setenv("LLM_ENGINE_SERVICE_SECRET", SECRET)
    from api.runner import RUNNER

    existing = set(REGISTRY._runs)
    RUNNER.set_paused(True)
    try:
        yield engine_client
    finally:
        for run_id in set(REGISTRY._runs) - existing:
            REGISTRY._runs.pop(run_id)
        RUNNER.set_paused(True)


def enqueue(kind: RunKind, status: RunStatus) -> str:
    """Create durable source-bearing fixtures without importing source or running custom code."""
    with store.session() as db:
        run = q.enqueue(db, kind=kind, label="job", config={"source": SOURCE}, status=status)
        run.result = {"graph": SOURCE}
        run.error = SOURCE
        run.stage = SOURCE
        return str(run.id)


async def test_private_validation_is_filtered_before_pagination_and_counts(client):
    """Hidden jobs cannot consume page slots, alter counts or appear in the active band."""
    public = [enqueue(RunKind.train, RunStatus.done) for _ in range(3)]
    for _ in range(4):
        enqueue(RunKind.model_validation, RunStatus.done)
    enqueue(RunKind.model_validation, RunStatus.queued)
    page = (await client.get("/runs?limit=2&visibility=administration", headers=headers(["public"]))).json()
    assert page["active"] == []
    assert page["history_total"] == 3
    assert len(page["history"]) == 2
    assert {value["run_id"] for value in page["history"]} <= set(public)
    assert SOURCE not in json.dumps(page)
    second = (await client.get("/runs?limit=2&offset=2", headers=headers(["public"]))).json()
    assert len(second["history"]) == 1
    admin = (await client.get("/runs", headers=headers(["administration"]))).json()
    assert admin["history_total"] == 7
    assert len(admin["active"]) == 1
    assert SOURCE in json.dumps(admin)


@pytest.mark.parametrize("scope", [None, [], ["public"], "administration", ["administration", "public"]])
async def test_private_detail_and_stream_deny_before_replay(client, scope):
    """Missing, malformed or conflicting scope never grants private row or live-registry access."""
    run_id = enqueue(RunKind.model_validation, RunStatus.running)
    live = REGISTRY.adopt(run_id, "model_validation", {"source": SOURCE})
    REGISTRY.publish(live, Event("validation", {"source": SOURCE}))
    for suffix in ("", "/events"):
        response = await client.get(f"/runs/{run_id}{suffix}?scope=administration", headers=headers(scope))
        assert response.status_code == 403
        assert SOURCE not in response.text
    assert not live.subscribers
    assert (await client.get(f"/runs/{run_id}", headers=headers(["administration"]))).json()["config"]["source"] == SOURCE


async def test_unsigned_administration_is_refused_when_service_auth_is_enforced(client):
    """A caller cannot forge the trusted carrier even with a syntactically valid admin scope."""
    run_id = enqueue(RunKind.model_validation, RunStatus.done)
    response = await client.get(f"/runs/{run_id}", headers=headers(["administration"], key="wrong-key"))
    assert response.status_code == 401


async def test_public_replay_omits_source_graph_logs_and_exception_text(client):
    """Both replay and terminal outcomes use a fixed public projection without cancelling durable work."""
    run_id = enqueue(RunKind.train, RunStatus.done)
    live = REGISTRY.adopt(run_id, "train", {})
    REGISTRY.publish(live, Event("log", {"line": SOURCE}))
    REGISTRY.publish(live, Event("train", {"loss": 1.25, "source": SOURCE, "graph": {"text": SOURCE}}))
    REGISTRY.finish(live, "failed", error=SOURCE)
    response = await client.get(f"/runs/{run_id}/events", headers=headers(["public"]))
    assert response.status_code == 200
    assert SOURCE not in response.text
    assert '"loss": 1.25' in response.text
    assert '"error": "Run failed"' in response.text
    assert not live.cancel.is_set()
    assert not live.subscribers


async def test_public_model_metadata_retains_summary_without_source_blobs(client):
    """Public list/detail/checkpoints expose measured facts while opaque admin blobs stay private."""
    with store.session() as db:
        model = q.upsert_model(db, model_id=uuid4(), name="published", published=True,
            context=128, parameters={"config": {"source": SOURCE}, "result": {"graph": SOURCE, "parameters": 123}})
        model_id = model.id
        q.upsert_model(db, model_id=uuid4(), name="checkpoint", parent_id=model_id,
            checkpoint_index=1, parameters={"source": SOURCE})
    for path in ("/models", f"/models/{model_id}", f"/models/{model_id}/checkpoints"):
        response = await client.get(path, headers=headers(["public"]))
        assert response.status_code == 200
        assert SOURCE not in response.text
    detail = (await client.get(f"/models/{model_id}", headers=headers(["public"])) ).json()
    assert detail["parameter_count"] == 123
    assert detail["context"] == 128
    admin = await client.get(f"/models/{model_id}", headers=headers(["administration"]))
    assert SOURCE in admin.text


@pytest.mark.parametrize("query", ["limit=101", "limit=0", "offset=-1"])
async def test_visibility_pages_keep_bounded_external_limits(client, query):
    """Invalid paging budgets are refused at the HTTP boundary before any query executes."""
    assert (await client.get(f"/runs?{query}", headers=headers(["public"])) ).status_code == 422
