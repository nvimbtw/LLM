"""Memory-bearing datasets retain structured context through normal chat cleanup."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

import processors
import store
from store import queries as q
from store.schema import DatasetKind, DatasetOrigin, DatasetStatus


@pytest.fixture
def memory_document() -> dict:
    """Build a raw SFT example from the shared project-memory wire contract."""
    contract = Path(__file__).resolve().parents[2] / "contracts" / "project-memory.json"
    request = json.loads(contract.read_text())["request"]
    return {
        "messages": [
            {"role": "project", "content": json.dumps(request["project"], indent=2)},
            {"role": "memory", "content": json.dumps(request["memories"][0], indent=2)},
            {"role": "human", "content": request["turns"][0]["text"]},
            {"role": "gpt", "content": "**Commit** the message before acknowledging it."},
        ],
        "meta": {"source": "project-memory-contract"},
    }


def cleanup_steps() -> list[dict]:
    """Return the existing plain-chat pipeline with explicit role and tool checks."""
    return [
        {"processor": "normalise-roles"},
        {"processor": "drop-tool-calls"},
        {"processor": "convert-chat"},
        {"processor": "markdown-emphasis"},
        {"processor": "require-roles"},
    ]


def test_chat_cleanup_preserves_structured_context_and_converts_answers(memory_document):
    """Context payload bytes and source metadata survive while replies gain markup."""
    before = deepcopy(memory_document)
    steps = processors.resolve(cleanup_steps(), processors.REGISTRY)
    cleaned = processors.run(steps, memory_document, {}).after

    assert cleaned is not None
    assert cleaned["messages"][:2] == before["messages"][:2]
    assert [turn["role"] for turn in cleaned["messages"]] == [
        "project", "memory", "user", "assistant"
    ]
    assert "<|bold|>Commit<|/bold|>" in cleaned["messages"][-1]["content"]
    assert cleaned["meta"] == before["meta"]
    assert memory_document == before


def test_past_calls_in_context_are_not_live_tool_use(memory_document):
    """A memory may quote a past call without making the current example tool-using."""
    memory = json.loads(memory_document["messages"][1]["content"])
    memory["memory"] = '<|tool-call|>{"tool":"bash","argument":"pwd"}<|/tool-call|>'
    memory_document["messages"][1]["content"] = json.dumps(memory)
    steps = processors.resolve(cleanup_steps(), processors.REGISTRY)

    cleaned = processors.run(steps, memory_document, {}).after

    assert cleaned is not None
    assert cleaned["messages"][1] == memory_document["messages"][1]
    assert json.loads(cleaned["messages"][1]["content"]) == memory


async def test_context_survives_dataset_processing_and_export(
    client, acting, owner, store_session, memory_document,
):
    """The stored, processed and exported SFT row keeps both context role payloads."""
    with store.session() as db:
        dataset = q.create_dataset(
            db, name="memory-source", kind=DatasetKind.sft,
            origin=DatasetOrigin.huggingface, owner_id=owner,
        )
        dataset_id = dataset.id
        q.add_documents(db, dataset_id, 0, [
            {**memory_document, "byte_size": len(json.dumps(memory_document).encode())}
        ])
        q.finish_dataset(db, dataset_id, DatasetStatus.ready)

    queued = await client.post(
        f"/datasets/{dataset_id}/process", headers=acting,
        json={"name": "memory-cleaned", "steps": cleanup_steps()},
    )
    assert queued.status_code == 202
    result = queued.json()
    for _ in range(200):
        run = (await client.get(f"/runs/{result['run_id']}", headers=acting)).json()
        if run["status"] in ("done", "failed", "cancelled"):
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("memory example processing did not settle")
    assert run["status"] == "done", run.get("error")

    page = await client.get(f"/datasets/{result['dataset_id']}/documents", headers=acting)
    assert page.status_code == 200
    stored = page.json()["documents"]
    assert len(stored) == 1
    assert stored[0]["meta"] == memory_document["meta"]
    exported = await client.get(f"/datasets/{result['dataset_id']}/export", headers=acting)
    assert exported.status_code == 200
    rows = [json.loads(line) for line in exported.text.splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["messages"][:2] == memory_document["messages"][:2]
