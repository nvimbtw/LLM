"""Signed, data-only selected-recipe preview and immutable lookup; synthetic acceptance fixtures do not certify numerical validity."""

from uuid import uuid4

import pytest
from sqlalchemy import func, select

import store
from llm.architecture.spec import canonical_digest
from llm.architecture.wire import ArchitectureDraftSave, MetadataPatch, TrainingArchitecturePreview
from store import architecture_queries as catalog
from store.schema import Run
from tests.test_authoring_metadata import forbid_execution
from tests.test_training_submission import client, final_tokenizer, sample, sessions
from tests.test_architecture_visibility import headers


def request(sample, **changes) -> dict:
    """Build an explicit preview request over a synthetic accepted revision without creating a training operation identity."""
    return {"selection": {"revision_id": str(sample.architecture_revision), "parameters": {}}, "batch_size": 3, **changes}


async def test_pinned_lookup_and_pending_preview_preserve_accepted_bindings_after_draft_edits(client, sample, monkeypatch):
    """Actual accepted metadata resolves sparse parameters/context without parsing the new invalid draft or creating a fake tokenizer/job."""
    forbid_execution(monkeypatch)
    admin = headers(["administration"])
    with store.session() as db:
        catalog.save_architecture_draft(db, sample.architecture_id, ArchitectureDraftSave(expected_version=1, yaml="new invalid yaml ["))
        before = db.scalar(select(func.count()).select_from(Run))
    root = f"/architectures/revisions/{sample.architecture_revision}"
    selected = await client.get(root, headers=admin)
    assert selected.status_code == 200, selected.text
    assert selected.json()["accepted"]["reference"]["resource_id"] == str(sample.architecture_id)
    assert selected.json()["yaml"] != "new invalid yaml ["
    response = await client.post(root + "/training-preview", headers=admin, json=request(sample))
    assert response.status_code == 200, response.text
    preview = TrainingArchitecturePreview.model_validate(response.json())
    assert preview.parameters == sample.parameters and preview.context_window == 8
    assert preview.parameter_digest == canonical_digest(sample.parameters)
    assert preview.tokenizer is None and preview.final_preflight_required is True
    assert preview.profile.domains[0].batch_max == 3 and preview.profile.domains[0].sequence_max == 8
    assert preview.profile.domains[0].batch_min == preview.profile.domains[0].sequence_min == 1
    assert not preview.profile.export_guards
    assert "receipt" not in response.json() and "run_id" not in response.json() and "build_request_id" not in response.text
    with store.session() as db:
        assert db.scalar(select(func.count()).select_from(Run)) == before


async def test_final_preview_uses_actual_owned_tokenizer_and_current_profile_controls(client, sample, monkeypatch):
    """Final bindings come from owned tokenizer bytes and changed numerical inputs remain prospective, never inferred receipts."""
    binding = final_tokenizer()
    forbid_execution(monkeypatch)
    response = await client.post(f"/architectures/revisions/{sample.architecture_revision}/training-preview", headers=headers(["administration"]),
        json=request(sample, tokenizer_id=str(binding.tokenizer_id), seed=7, batch_size=2, precision="fp32", allow_tf32=False))
    assert response.status_code == 200, response.text
    preview = TrainingArchitecturePreview.model_validate(response.json())
    assert preview.tokenizer.tokenizer_id == binding.tokenizer_id and preview.tokenizer.fingerprint == binding.fingerprint
    assert preview.profile.numerical_seed == 7 and preview.profile.domains[0].batch_max == 2
    assert preview.final_preflight_required


@pytest.mark.parametrize("scope", [None, [], ["public"], "administration", ["administration", "public"]])
async def test_selection_lookup_and_preview_require_exact_signed_administrator_scope(client, sample, monkeypatch, scope):
    """Ordinary or malformed signed visibility cannot read selected source or trigger configuration/tokenizer artifact reads."""
    forbid_execution(monkeypatch)
    root = f"/architectures/revisions/{sample.architecture_revision}"
    assert (await client.get(root, headers=headers(scope))).status_code == 403
    assert (await client.post(root + "/training-preview", headers=headers(scope), json=request(sample))).status_code == 403


async def test_preview_refuses_foreign_revision_archived_new_selection_and_invalid_inputs(client, sample):
    """Preview failures remain bounded selection errors and immutable history stays readable after archive."""
    admin = headers(["administration"])
    root = f"/architectures/revisions/{sample.architecture_revision}"
    assert (await client.get(f"/architectures/revisions/{uuid4()}", headers=admin)).status_code == 404
    assert (await client.post(root + "/training-preview", headers=admin, json={**request(sample), "selection": {"revision_id": str(uuid4()), "parameters": {}}})).status_code == 422
    assert (await client.post(root + "/training-preview", headers=admin, json=request(sample, batch_size=0))).status_code == 422
    with store.session() as db:
        catalog.update_metadata(db, "architecture", sample.architecture_id, MetadataPatch(expected_metadata_version=1, archived=True))
    assert (await client.post(root + "/training-preview", headers=admin, json=request(sample))).status_code == 409
    assert (await client.get(root, headers=admin)).status_code == 200
