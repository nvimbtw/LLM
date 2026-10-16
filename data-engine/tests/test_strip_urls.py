"""The proof file is discovered, maps both document shapes and previews stored rows."""

import pytest

import processors
import store
from store import queries as q
from store.schema import DatasetKind, DatasetOrigin, DatasetStatus


@pytest.mark.parametrize(
    "text,expected,min_len",
    [
        ("see http://a.test https://b.test www.c.test", "see   ", 0),
        ("https://only.test", None, 0),
        ("  ", None, 0),
        ("short", None, 6),
        ("short", "short", 5),
    ],
)
def test_strip_urls_rewrites_or_drops(text, expected, min_len):
    """Keep exact remaining whitespace and enforce empty/minimum-length policy."""
    entry = processors.REGISTRY["strip-urls"]
    result = entry.apply({"content": text}, entry.resolve({"min_len": min_len}), {})
    assert (result["content"] if result else None) == expected


def test_strip_urls_drops_a_conversation_if_one_turn_becomes_empty():
    """Dual-kind mapping applies independently to every turn and retains exchange integrity."""
    entry = processors.REGISTRY["strip-urls"]
    doc = {
        "messages": [
            {"role": "user", "content": "https://only.test"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    assert entry.apply(doc, entry.defaults(), {}) is None


async def test_strip_urls_preview_reports_drops_from_stored_documents(
    client, acting, owner
):
    """The file-drop catalogue and preview route run the same registered callback."""
    with store.session() as db:
        row = q.create_dataset(
            db,
            name="url-preview",
            kind=DatasetKind.pretrain,
            origin=DatasetOrigin.huggingface,
            owner_id=owner,
        )
        identity = row.id
        q.add_documents(
            db,
            identity,
            0,
            [
                {"content": "Read https://example.test now"},
                {"content": "www.example.test"},
            ],
        )
        q.finish_dataset(db, identity, DatasetStatus.ready)
    response = await client.post(
        "/preview",
        headers=acting,
        json={"dataset_id": str(identity), "steps": [{"processor": "strip-urls"}]},
    )
    body = response.json()
    assert body["sampled"] == 2 and body["dropped"] == 1
    assert body["results"][0]["after"]["content"] == "Read  now"
    assert body["results"][1]["dropped_by"] == "strip-urls"
    assert len(body["steps"][0]["executed_version_hash"]) == 64


async def test_preview_reports_current_hash_instead_of_client_version(
    client, acting, pretrain_dataset
):
    """Preview always describes its active callback, even when sent an older recipe hash."""
    response = await client.post(
        "/preview",
        headers=acting,
        json={
            "dataset_id": str(pretrain_dataset),
            "steps": [{"processor": "strip-urls", "version_hash": "0" * 64}],
        },
    )
    record = response.json()["steps"][0]
    assert record["version_hash"] == processors.REGISTRY["strip-urls"].version_hash
    assert record["executed_version_hash"] == record["version_hash"]
    assert record["version_mismatch"] is False
