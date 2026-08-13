"""Every engine route: its shape, its status codes, and its error paths.

Replaces `test_routes.sh`, which needed a running server and a human reading
truncated JSON to decide whether anything had changed.
"""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from tests.fixtures import trace_spec
from tests.fixtures.tiny_env import TINY_DATASET, TINY_MODEL

UNKNOWN = "no-such-model"


@pytest.fixture
async def client(data_root):
    import main
    from llm.server.runs import REGISTRY

    # `main` binds this in an on_event("startup") hook, which ASGITransport
    # never fires — worker threads publish progress onto the loop it names.
    REGISTRY.bind_loop(asyncio.get_running_loop())

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://engine") as c:
        yield c


async def test_health_reports_ok_and_no_active_run(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "active_run": None}


async def test_datasets_lists_the_input_tree(client):
    body = (await client.get("/datasets")).json()
    assert [d["name"] for d in body["datasets"]] == [TINY_DATASET]


async def test_sft_datasets_lists_the_sft_tree(client):
    body = (await client.get("/sft/datasets")).json()
    assert [d["name"] for d in body["datasets"]] == [TINY_DATASET]


async def test_models_lists_weights_on_disk(client):
    """A subset assertion on purpose: other tests train models into the same
    tree, and the route's contract is "what is on disk", not "only this"."""
    body = (await client.get("/models")).json()
    listed = {m["name"]: m for m in body["models"]}
    assert TINY_MODEL in listed
    assert listed[TINY_MODEL]["has_weights"] is True


async def test_tokenize_count_matches_the_recorded_trace(client, recorded_trace):
    response = await client.post(
        "/tokenize/count", json={"model_name": TINY_MODEL, "text": trace_spec.QUESTION}
    )
    assert response.status_code == 200
    assert response.json()["tokens"] == len(recorded_trace["question_ids"])


async def test_tokenize_count_404s_for_an_unknown_model(client):
    response = await client.post(
        "/tokenize/count", json={"model_name": UNKNOWN, "text": "hi"}
    )
    assert response.status_code == 404


async def test_chat_returns_the_recorded_reply(client, recorded_trace):
    response = await client.post("/chat", json=_chat_body())
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == recorded_trace["sampled"]["text"]
    assert body["prompt_tokens"] == len(recorded_trace["prompt_ids"])


async def test_chat_404s_for_an_unknown_model(client):
    response = await client.post("/chat", json=_chat_body(model=UNKNOWN))
    assert response.status_code == 404


async def test_chat_stream_frames_concatenate_to_the_same_reply(client, recorded_trace):
    async with client.stream("POST", "/chat/stream", json=_chat_body()) as response:
        assert response.status_code == 200
        payload = "".join([chunk async for chunk in response.aiter_text()])

    events = _sse(payload)
    tokens = "".join(e["data"]["token"] for e in events if e["event"] == "token")
    done = [e for e in events if e["event"] == "done"]

    assert tokens == recorded_trace["sampled"]["text"]
    assert len(done) == 1
    assert done[0]["data"]["text"] == recorded_trace["sampled"]["text"]


async def test_runs_is_a_bare_list(client):
    """Unlike /models and /datasets, this is not wrapped in a key."""
    assert (await client.get("/runs")).json() == []


async def test_unknown_run_404s(client):
    assert (await client.get("/runs/nope")).status_code == 404
    assert (await client.post("/runs/nope/cancel")).status_code == 404


async def test_train_runs_end_to_end_and_writes_a_model(client):
    """The whole pipeline: sources, tokenizer, pack, loop, save."""
    response = await client.post(
        "/train?wait=true",
        json={
            "model_name": "trained",
            "datasets": [TINY_DATASET],
            "dimensions": 32,
            "heads": 4,
            "layers": 2,
            "ff_multiplier": 4,
            "ctx_window": 64,
            "max_vocab": 320,
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.001,
        },
    )
    assert response.status_code == 202
    snapshot = response.json()
    assert snapshot["status"] == "completed", snapshot.get("error")

    listed = (await client.get("/models")).json()["models"]
    assert "trained" in [m["name"] for m in listed]


async def test_train_rejects_a_second_run_while_one_holds_the_device(client, monkeypatch):
    """One GPU, one line — the invariant the whole queue design rests on."""
    from llm.server.runs import REGISTRY

    started = REGISTRY.start("train", {}, lambda on_progress, cancel: _block(cancel))
    try:
        response = await client.post("/train", json={"model_name": "second"})
        assert response.status_code == 409
    finally:
        started.cancel.set()
        await REGISTRY.wait(started)


def _block(cancel):
    cancel.wait(timeout=5)
    return {}


def _chat_body(model: str = TINY_MODEL) -> dict:
    return {
        "model_name": model,
        "turns": [{"role": "user", "text": trace_spec.QUESTION}],
        "system": trace_spec.SYSTEM,
        "max_tokens": trace_spec.MAX_TOKENS,
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.95,
        "seed": trace_spec.SEED,
    }


def _sse(payload: str) -> list[dict]:
    """Parse the `event:`/`data:` pairs SSE delimits with a blank line."""
    import json

    events = []
    for block in payload.split("\n\n"):
        if not block.strip():
            continue
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if event is not None:
            events.append({"event": event, "data": data})
    return events
