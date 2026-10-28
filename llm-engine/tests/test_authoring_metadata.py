"""Actual signed data-only metadata/declaration routes, separate from numerical validation and queue execution."""

from uuid import uuid4

import pytest
import torch
import yaml
from sqlalchemy import select

import store
from llm.architecture.builtins.recipes import recipe_template
from llm.architecture.wire import ArchitectureSchema, RecipeDraftInspection
from store.schema import Run
from tests.test_architecture_catalog_routes import client
from tests.test_architecture_visibility import headers


def forbid_execution(monkeypatch):
    """Refuse model construction, source execution or tensor loading during explicitly data-only declaration reads."""
    def forbidden(*args, **kwargs):
        """Fail if a metadata request reaches any numerical/custom-code entry point."""
        pytest.fail("metadata readers cannot execute custom source or construct/load models")
    monkeypatch.setattr("llm.architecture.source.SourceLoader.load", forbidden)
    monkeypatch.setattr("llm.architecture.build.build_model", forbidden)
    monkeypatch.setattr("torch.load", forbidden)


async def saved_recipe(client):
    """Create real current pinned YAML bytes as a draft, without accepting or executing its dependencies."""
    admin = headers(["administration"])
    source = yaml.safe_dump(recipe_template("looped").model_dump(mode="json", exclude_none=True), sort_keys=False)
    response = await client.post("/architectures", headers=admin, json={"name": "Draft declarations", "yaml": source})
    assert response.status_code == 201
    identity = response.json()["resource_id"]
    view = (await client.get(f"/architectures/{identity}/draft", headers=admin)).json()
    subject = {"kind": "draft", "resource_id": identity, "draft_version": view["draft_version"], "content_hash": view["content_hash"]}
    return identity, subject, source


async def test_schema_reports_installed_runtime_defaults_without_numerical_authorization(client, monkeypatch):
    """Real schema JSON includes a prospective installed runtime/profile and authoring limits, never a validation receipt."""
    forbid_execution(monkeypatch)
    response = await client.get("/architectures/schema", headers=headers(["administration"]))
    assert response.status_code == 200
    metadata = ArchitectureSchema.model_validate(response.json())
    assert metadata.recipe["title"] == "Recipe"
    assert metadata.default_profile.torch_version == str(torch.__version__)
    assert metadata.default_profile.device == "cpu" and "cpu" in metadata.available_devices
    assert metadata.limits.source_bytes == 262144 and metadata.default_profile.domains[0].sequence_max == 8
    assert "receipt" not in response.json()


async def test_exact_saved_yaml_declarations_never_resolve_import_or_enqueue_custom_code(client, monkeypatch):
    """Actual declared controls/recipe references remain readable without any constructed model or queued validation claim."""
    identity, subject, source = await saved_recipe(client)
    forbid_execution(monkeypatch)
    admin = headers(["administration"])
    response = await client.post(f"/architectures/{identity}/declaration", headers=admin, json=subject)
    assert response.status_code == 200
    metadata = RecipeDraftInspection.model_validate(response.json())
    assert metadata.subject.model_dump(mode="json") == subject and metadata.verdict == "inspection_only"
    assert metadata.recipe == recipe_template("looped")
    assert "dimensions" in metadata.recipe.parameters and metadata.recipe.modules
    assert set(response.json()) == {"subject", "recipe", "verdict"}
    assert (await client.get(f"/architectures/{identity}/draft", headers=admin)).json()["yaml"] == source
    with store.session() as db:
        assert list(db.scalars(select(Run))) == []


async def test_changed_or_foreign_draft_identity_cannot_return_a_current_declaration(client):
    """Stale draft versions/hashes and another resource's subject refuse before parsing replacement bytes."""
    identity, subject, _ = await saved_recipe(client)
    root = f"/architectures/{identity}"
    admin = headers(["administration"])
    changed = await client.put(root + "/draft", headers=admin, json={"expected_version": 1, "yaml": "invalid ["})
    assert changed.status_code == 200
    stale = await client.post(root + "/declaration", headers=admin, json=subject)
    assert stale.status_code == 409 and stale.json()["current_version"] == 2
    assert stale.json()["error"]["phase"] == "inspection"
    foreign = await client.post(root + "/declaration", headers=admin, json={**subject, "resource_id": str(uuid4())})
    assert foreign.status_code == 422 and foreign.json()["error"]["code"] == "invalid_subject"
    current = {**subject, "draft_version": 2, "content_hash": changed.json()["content_hash"]}
    invalid = await client.post(root + "/declaration", headers=admin, json=current)
    assert invalid.status_code == 422 and invalid.json()["error"]["phase"] == "inspection"


@pytest.mark.parametrize("scope", [None, ["public"], ["administration", "public"]])
async def test_declaration_metadata_requires_exact_administrative_actor_scope(client, scope):
    """Neither schema nor explicitly parsed source declarations are available under public or mixed service visibility."""
    identity, subject, _ = await saved_recipe(client)
    assert (await client.get("/architectures/schema", headers=headers(scope))).status_code == 403
    response = await client.post(f"/architectures/{identity}/declaration", headers=headers(scope), json=subject)
    assert response.status_code == 403
