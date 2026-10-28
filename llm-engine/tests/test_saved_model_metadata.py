"""Saved model reads use immutable metadata without source/tensor execution or live catalog dependencies."""

import json
import hashlib
from uuid import uuid4

import pytest

import store
from llm import storage
from llm.architecture import package
from llm.architecture.package_metadata import read_saved_metadata
from llm.architecture.spec import DefinitionPackage
from llm.architecture.wire import GraphQuery, ModelDefinitionSummary, RetainedSource
from store import queries as q
from tests.package_fixtures import payload
from tests.test_architecture_catalog_routes import client
from tests.test_architecture_visibility import headers


@pytest.fixture
def saved_model(data_root, store_session):
    """Publish an explicitly synthetic nested/shared custom package; metadata tests claim no numerical validation evidence."""
    fixture = payload("nested", custom=True)
    manifest = DefinitionPackage.model_validate(fixture["manifest"])
    files = {name: value.encode() for name, value in fixture["files"].items()}
    files.update({"model.pt": b"synthetic tensors", "train_state.pt": b"synthetic optimizer"})
    model_id = uuid4()
    directory = storage.model_dir(str(model_id))
    package.publish(directory, manifest, files)
    with store.session() as db:
        q.upsert_model(db, model_id=model_id, name="saved nested model", published=True, parameters={"config": {
            "dimensions": 999, "layers": 999, "heads": 999, "ff_multiplier": 999, "ctx_window": 999}})
    return model_id, directory, manifest


def forbidden_execution(*args, **kwargs):
    """Detect any source import, model construction, tensor decoding or recapture during a metadata-only read."""
    pytest.fail("saved metadata read attempted numerical/source execution")


def disable_execution(monkeypatch):
    """Forbid every execution entrance while leaving safe data parsing and bounded metadata reads available."""
    monkeypatch.setattr("llm.architecture.source.SourceLoader.load", forbidden_execution)
    monkeypatch.setattr("llm.architecture.build.build_model", forbidden_execution)
    monkeypatch.setattr("llm.architecture.validation.validate_snapshot", forbidden_execution)
    monkeypatch.setattr("torch.load", forbidden_execution)


async def test_saved_definition_summary_and_graph_survive_without_catalog_or_tensor_files(client, saved_model, monkeypatch):
    """Admin reads retain exact configured semantics/receipt identity and never touch source, tensor or optimizer payloads."""
    disable_execution(monkeypatch)
    model_id, directory, manifest = saved_model
    (directory / "model.pt").unlink()
    (directory / "train_state.pt").unlink()
    for dependency in manifest.definition.dependencies:
        (directory / dependency.path).unlink()
    response = await client.get(f"/models/{model_id}/definition", headers=headers(["administration"]))
    assert response.status_code == 200, response.text
    summary = ModelDefinitionSummary.model_validate(response.json())
    assert summary.definition_id == manifest.definition_id
    assert summary.architecture == manifest.definition.architecture
    assert summary.parameters == manifest.definition.parameters
    assert summary.parameter_count == 123 and summary.legacy_architecture is None
    assert summary.context_window == 12 and summary.vocabulary_size == 24
    assert len(summary.validation_receipts) == 1
    query = GraphQuery(identity_hash=summary.validation_receipts[0].identity_hash,
        view="operations", mode="eval", signature="full_prefix", limit=1)
    graph = await client.get(f"/models/{model_id}/graph", params=query.model_dump(exclude_none=True), headers=headers(["administration"]))
    assert graph.status_code == 200, graph.text
    assert graph.json()["nodes"][0]["label"] == "Synthetic package fixture"
    detail = (await client.get(f"/models/{model_id}", headers=headers(["public"]))).json()
    assert detail["context"] == 12 and detail["vocab_size"] == 24 and detail["parameter_count"] == 123
    assert detail["architecture"] is None
    assert detail["definition_id"] == manifest.definition_id
    assert detail["context_extension_supported"] is False
    assert "validation_receipts" not in detail and detail["parameters"] is None


@pytest.mark.parametrize("scope", [None, [], ["public"], "administration", ["administration", "public"]])
async def test_saved_definition_and_graph_require_exact_admin_before_reading_artifacts(client, saved_model, monkeypatch, scope):
    """Model publication does not expose source/graph administration through browser-supplied scope arguments."""
    model_id, _, _ = saved_model
    monkeypatch.setattr("api.routes.model_definitions.read_saved_metadata", forbidden_execution)
    for path in ("definition", f"sources/{uuid4()}", "graph?identity_hash=" + "a" * 64 + "&view=operations&mode=eval&signature=full_prefix"):
        response = await client.get(f"/models/{model_id}/{path}", headers=headers(scope))
        assert response.status_code == 403, response.text


async def test_graph_rejects_stale_identity_unsupported_scope_and_corrupt_saved_bytes(client, saved_model):
    """Old identities cannot select another profile and bytes changed after publication never become valid graph output."""
    model_id, directory, manifest = saved_model
    saved = read_saved_metadata(directory)
    query = {"identity_hash": saved.receipts[0].identity_hash, "view": "operations", "mode": "eval", "signature": "full_prefix"}
    admin = headers(["administration"])
    path = f"/models/{model_id}/graph"
    stale = await client.get(path, headers=admin, params={**query, "identity_hash": "a" * 64})
    assert stale.status_code == 409
    unsupported = await client.get(path, headers=admin, params={**query, "scope": "unknown"})
    assert unsupported.status_code == 422
    graph_manifest = json.loads((directory / f"graphs/{saved.receipts[0].graph_manifest_hash}.json").read_text())
    selected = next(entry for entry in graph_manifest["graphs"] if entry["mode"] == "eval")
    (directory / f"graphs/{selected['sha256']}.json").write_text("{}")
    corrupt = await client.get(path, headers=admin, params=query)
    assert corrupt.status_code == 409


async def test_legacy_missing_and_corrupt_definition_reads_return_bounded_errors(client, saved_model):
    """Missing definitions and corrupted persisted metadata fail without falling back to model construction."""
    model_id, directory, _ = saved_model
    admin = headers(["administration"])
    assert (await client.get(f"/models/{uuid4()}/definition", headers=admin)).status_code == 404
    (directory / "meta.json").write_text('{"result":{"parameters":999}}')
    response = await client.get(f"/models/{model_id}/definition", headers=admin)
    assert response.status_code == 409 and str(directory) not in response.text


async def test_saved_sources_are_exact_selected_bytes_without_catalog_or_execution(client, saved_model, monkeypatch):
    """Imported roots/custom dependencies/builtin descriptors survive missing tensor files and never substitute installed code."""
    disable_execution(monkeypatch)
    model_id, directory, manifest = saved_model
    (directory / "model.pt").unlink()
    (directory / "train_state.pt").unlink()
    admin = headers(["administration"])
    entries = [(manifest.definition.architecture, "architecture.yaml")]
    entries.extend((entry.reference, entry.path) for entry in manifest.definition.dependencies)
    for reference, path in entries:
        response = await client.get(f"/models/{model_id}/sources/{reference.revision_id}", headers=admin)
        assert response.status_code == 200, response.text
        source = RetainedSource.model_validate(response.json())
        assert source.source.encode() == (directory / path).read_bytes()
        assert source.definition_id == manifest.definition_id
        assert source.content_hash == reference.content_hash
        assert source.language == ("json" if path.endswith(".builtin.json") else "yaml" if reference.kind == "architecture" else "python")
    assert (await client.get(f"/models/{model_id}/sources/{uuid4()}", headers=admin)).status_code == 404


async def test_saved_source_corruption_and_missing_bytes_refuse_without_fallback(client, saved_model, monkeypatch):
    """Integrity and semantic revision hashes both gate reads; missing or linked source cannot resolve from live catalog/installed files."""
    disable_execution(monkeypatch)
    model_id, directory, manifest = saved_model
    entry = next(entry for entry in manifest.definition.dependencies if entry.path.endswith(".py"))
    path = directory / entry.path
    endpoint = f"/models/{model_id}/sources/{entry.reference.revision_id}"
    admin = headers(["administration"])
    path.write_text("tampered source")
    response = await client.get(endpoint, headers=admin)
    assert response.status_code == 409 and "tampered source" not in response.text
    inventory = json.loads((directory / "integrity.json").read_text())
    inventory["files"][entry.path] = {"size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory / "integrity.json").write_text(json.dumps(inventory))
    assert (await client.get(endpoint, headers=admin)).status_code == 409
    path.unlink()
    assert (await client.get(endpoint, headers=admin)).status_code == 409
    path.symlink_to(directory / "architecture.yaml")
    assert (await client.get(endpoint, headers=admin)).status_code == 409
