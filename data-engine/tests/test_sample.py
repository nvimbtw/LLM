"""Looking inside a dataset before pulling it.

The network is stubbed at `httpx.get`: these tests are about what the service
makes of the server's answer, and a test that needs the Hub online is a test
nobody runs.
"""

import json

import pytest


class _Response:
    def __init__(self, payload, status: int = 200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("boom", request=None, response=self)


SPLITS = {"splits": [
    {"config": "default", "split": "test"},
    {"config": "default", "split": "train"},
]}

PROSE_ROWS = {
    "features": [{"name": "text"}, {"name": "url"}],
    "rows": [
        {"row": {"text": "Some prose, at length.", "url": "https://example.test/a"}},
        {"row": {"text": "More of it.", "url": "https://example.test/b"}},
    ],
}

CHAT_ROWS = {
    "features": [{"name": "conversations"}],
    "rows": [
        {"row": {"conversations": [
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "hi"},
        ]}},
    ],
}


@pytest.fixture
def hub_server(monkeypatch):
    """Answers `/splits` and `/first-rows`, recording what it was asked."""
    calls: list[dict] = []

    def serve(first_rows):
        def get(url, params=None, timeout=None):
            calls.append({"url": url, "params": dict(params or {})})
            if url.endswith("/splits"):
                return _Response(SPLITS)
            return _Response(first_rows)

        import httpx

        # Patched on `httpx` itself: `hub.sample` the name resolves to the
        # re-exported function, not the module that calls it.
        monkeypatch.setattr(httpx, "get", get)
        return calls

    return serve


def test_prose_rows_suggest_a_pretrain_pull(hub_server):
    hub_server(PROSE_ROWS)
    import hub

    found = hub.sample("owner/corpus")

    assert found["suggested_kind"] == "pretrain"
    assert found["suggested_column"] == "text"
    assert found["columns"] == ["text", "url"]
    assert found["rows"][0]["text"].startswith("Some prose")


def test_turn_shaped_rows_suggest_an_sft_pull(hub_server):
    hub_server(CHAT_ROWS)
    import hub

    found = hub.sample("owner/chat")

    assert found["suggested_kind"] == "sft"
    assert found["suggested_column"] == "conversations"
    # Roles arrive as the source wrote them; normalising is a processor's job.
    assert found["rows"][0]["conversations"][0]["from"] == "human"


def test_train_is_preferred_when_no_split_is_named(hub_server):
    calls = hub_server(PROSE_ROWS)
    import hub

    found = hub.sample("owner/corpus")

    assert found["split"] == "train"
    assert calls[-1]["params"]["split"] == "train"
    # Every indexed pair comes back, so the caller can offer the others without
    # asking again.
    assert {s["split"] for s in found["splits"]} == {"train", "test"}


def test_a_named_split_is_honoured(hub_server):
    hub_server(PROSE_ROWS)
    import hub

    assert hub.sample("owner/corpus", "default", "test")["split"] == "test"


def test_an_unknown_split_falls_back_rather_than_failing(hub_server):
    """It usually means a config carried over from the repo looked at before."""
    hub_server(PROSE_ROWS)
    import hub

    assert hub.sample("owner/corpus", "nonexistent", "nonexistent")["split"] == "train"


def test_long_cells_are_cut_and_the_answer_says_so(hub_server):
    from hub.sample import MAX_CELL_CHARS

    hub_server(
        {
            "features": [{"name": "text"}],
            "rows": [{"row": {"text": "x" * (MAX_CELL_CHARS * 2)}}],
        }
    )
    import hub

    found = hub.sample("owner/corpus")

    assert len(found["rows"][0]["text"]) == MAX_CELL_CHARS
    assert found["truncated"] is True


def test_the_row_cap_is_enforced_however_many_arrive(hub_server):
    hub_server(PROSE_ROWS)
    import hub

    assert len(hub.sample("owner/corpus", rows=1)["rows"]) == 1


def test_a_server_refusal_carries_its_reason(monkeypatch):
    import httpx

    import hub

    def get(url, params=None, timeout=None):
        return _Response({"error": "dataset is gated"}, status=403)

    monkeypatch.setattr(httpx, "get", get)

    with pytest.raises(hub.HubError, match="gated"):
        hub.sample("owner/private")


async def test_the_route_answers_a_sample(client, acting, hub_server):
    hub_server(PROSE_ROWS)

    response = await client.get("/hub/datasets/sample?repo=owner/corpus", headers=acting)

    assert response.status_code == 200
    assert response.json()["suggested_column"] == "text"


async def test_the_route_reports_a_hub_failure_as_a_bad_gateway(client, acting, monkeypatch):
    """502, not 500: the Hub failed, not this service."""
    import httpx

    def get(url, params=None, timeout=None):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(httpx, "get", get)

    response = await client.get("/hub/datasets/sample?repo=owner/corpus", headers=acting)

    assert response.status_code == 502
