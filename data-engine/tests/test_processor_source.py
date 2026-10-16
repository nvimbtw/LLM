"""Source API separates saved bytes, active code and load errors."""

import hashlib
import json
from pathlib import Path

import pytest

import processors
from processors import loader
from processors.loader import Catalog, MAX_SOURCE_BYTES
from tests.test_processor_loader import SOURCE

CONTRACT = json.loads(
    (
        Path(__file__).resolve().parents[2] / "contracts/processor-source.json"
    ).read_text()
)


@pytest.fixture
def custom_catalog(tmp_path, monkeypatch):
    """Install an isolated custom catalogue without editing checked-in processors."""
    (tmp_path / "example.py").write_text(SOURCE)
    catalog = Catalog(dict(processors.REGISTRY.snapshot()), tmp_path)
    monkeypatch.setattr(processors, "REGISTRY", catalog)
    return catalog


async def test_saved_edit_reports_global_reload_failure(
    client, acting, custom_catalog, monkeypatch
):
    """A rejected registry publication still reports that source was saved on disk."""
    monkeypatch.setattr(loader, "MAX_CUSTOM_FILES", 1)
    (custom_catalog.directory / "extra.py").write_text(
        SOURCE.replace('name="example"', 'name="extra"')
    )
    previous_hash = custom_catalog["example"].version_hash
    changed = SOURCE.replace("text.upper()", "text.lower()")
    saved = (
        await client.put(
            "/processors/example/source", headers=acting, json={"source": changed}
        )
    ).json()
    assert saved["saved"] is True
    assert saved["active_version_hash"] == previous_hash != saved["version_hash"]
    assert "at most 1" in saved["reload_error"]


async def test_create_processor_activates_valid_source_without_overwriting_existing_files(
    client, acting, custom_catalog
):
    """A new operator definition is immediately viewable, editable and in the catalogue."""
    text = SOURCE.replace('name="example"', 'name="created"')
    response = await client.post("/processors", headers=acting, json={"source": text})
    assert response.status_code == 201
    info = response.json()
    assert info["name"] == "created" and info["editable"] is True
    assert info["source"] == text
    assert info["active_version_hash"] == info["version_hash"]
    path = custom_catalog.directory / "created.py"
    assert path.read_text() == text
    assert (
        custom_catalog["created"].apply({"content": "hello"}, {}, {})["content"]
        == "HELLO"
    )
    again = await client.post(
        "/processors",
        headers=acting,
        json={"source": text.replace("upper()", "lower()")},
    )
    assert again.status_code == 409
    assert path.read_text() == text


@pytest.mark.parametrize(
    "text",
    [
        "def broken(",
        SOURCE.replace('name="example"', 'name="new-processor"'),
        SOURCE.replace('name="example"', 'name="../escape"'),
        SOURCE.replace('title="Example"', "title={}"),
    ],
)
async def test_invalid_first_versions_do_not_create_files(
    client, acting, custom_catalog, text
):
    """Creation validates the first working version and never leaves a failed new file."""
    before = set(custom_catalog.directory.iterdir())
    response = await client.post("/processors", headers=acting, json={"source": text})
    assert response.status_code == 422
    assert set(custom_catalog.directory.iterdir()) == before


async def test_source_save_roundtrip_and_active_catalogue_hash(
    client, acting, custom_catalog
):
    """A valid edit updates disk, catalogue and subsequent preview metadata together."""
    before = (await client.get("/processors/example/source", headers=acting)).json()
    permissions = (custom_catalog.directory / "example.py").stat().st_mode
    assert before["source"] == SOURCE
    assert before["editable"] is True
    changed = SOURCE.replace("text.upper()", "text.lower()")
    response = await client.put(
        "/processors/example/source", headers=acting, json={"source": changed}
    )
    assert response.status_code == 200
    saved = response.json()
    assert saved["saved"] is True
    assert saved["reload_error"] is None
    assert (custom_catalog.directory / "example.py").stat().st_mode == permissions
    assert saved["version_hash"] == hashlib.sha256(changed.encode()).hexdigest()
    assert (
        saved["active_version_hash"] == saved["version_hash"] != before["version_hash"]
    )
    listed = (await client.get("/processors", headers=acting)).json()["processors"]
    assert (
        next(p for p in listed if p["name"] == "example")["version_hash"]
        == saved["version_hash"]
    )


async def test_invalid_save_persists_bytes_and_preview_uses_old_version(
    client, acting, custom_catalog, pretrain_dataset
):
    """Saved source errors do not change the active callback or its preview hash."""
    active_hash = custom_catalog["example"].version_hash
    response = await client.put(
        "/processors/example/source", headers=acting, json=CONTRACT["request"]
    )
    saved = response.json()
    assert saved["saved"] is True
    assert "SyntaxError" in saved["reload_error"]
    assert {
        **saved,
        "reload_error": saved["reload_error"].split(":", 1)[0],
    } == CONTRACT["save_response"]
    info = (await client.get("/processors/example/source", headers=acting)).json()
    assert info["source"] == "def broken("
    assert info["version_hash"] != info["active_version_hash"] == active_hash
    assert {**info, "reload_error": info["reload_error"].split(":", 1)[0]} == CONTRACT[
        "source_response"
    ]
    catalogue = (await client.get("/processors", headers=acting)).json()["processors"]
    assert {"processors": [p for p in catalogue if p["name"] == "example"]} == CONTRACT[
        "catalogue"
    ]
    preview = (
        await client.post(
            "/preview",
            headers=acting,
            json={
                "dataset_id": str(pretrain_dataset),
                "steps": [{"processor": "example"}],
                "limit": 1,
            },
        )
    ).json()
    assert preview["steps"][0]["executed_version_hash"] == active_hash
    assert "HELLO" in preview["results"][0]["after"]["content"]
    assert not list(custom_catalog.directory.glob("*.tmp"))


async def test_unknown_sources_builtin_writes_and_byte_budget(
    client, acting, custom_catalog
):
    """Unknown identities are 404; builtins are read-only; UTF-8 source has a byte ceiling."""
    assert (
        await client.get("/processors/missing/source", headers=acting)
    ).status_code == 404
    assert (
        await client.put(
            "/processors/missing/source", headers=acting, json={"source": ""}
        )
    ).status_code == 404
    builtin = (await client.get("/processors/strip-html/source", headers=acting)).json()
    assert builtin["editable"] is False
    assert (
        await client.put(
            "/processors/strip-html/source", headers=acting, json={"source": ""}
        )
    ).status_code == 403
    assert (
        await client.put(
            "/processors/example/source",
            headers=acting,
            json={"source": "é" * (MAX_SOURCE_BYTES // 2 + 1)},
        )
    ).status_code == 422
