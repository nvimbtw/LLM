"""Builtin bootstrap and real numerical fixture publication; actual production worker acceptance remains separate."""

import pytest
from sqlalchemy import select

import store
from api import builtin_catalog
from llm.architecture.builtins.catalog import builtin_definitions, readable_source
from llm.architecture.source import SourceLoader
from llm.architecture.validation import validate_snapshot
from llm.architecture.spec import RevisionSubject
from store.schema import BlockRevision, ModelBlock, Run, RunKind
from store import architecture_queries as catalog
from tests.test_validation_queue import client, publish_fixture, queued_snapshot
from tests.test_architecture_visibility import headers


def test_builtin_startup_seeds_only_readable_drafts_and_idempotent_normal_queue_jobs(store_session,monkeypatch):
    """Seeding constructs no model/imports no custom source and cannot manufacture acceptance evidence."""
    def forbidden(*_args,**_kwargs):
        """Fail if a data-only bootstrap attempts model construction or custom import."""
        pytest.fail("builtin startup must not execute numerical model code")
    monkeypatch.setattr(SourceLoader,"load",forbidden)
    monkeypatch.setattr("torch.nn.Module.__init__",forbidden)
    builtin_catalog.seed_blocks()
    builtin_catalog.seed_blocks()
    with store.session() as db:
        roots = list(db.scalars(select(ModelBlock)))
        runs = list(db.scalars(select(Run)))
        assert len(roots) == len(runs) == 9
        assert all(row.origin == "builtin" and row.draft_version == 1 for row in roots)
        assert not list(db.scalars(select(BlockRevision)))
        assert all(row.kind == RunKind.model_validation and row.owner_id == builtin_catalog.BUILTIN_ACTOR for row in runs)


def test_builtin_validation_refuses_changed_readable_source_before_construction(store_session,monkeypatch):
    """A valid implementation digest cannot authorize different supplied Python bytes or fall back to custom import."""
    import hashlib

    builtin_catalog.seed_blocks()
    with store.session() as db:
        run_id = str(next(iter(db.scalars(select(Run)))).id)
    snapshot = queued_snapshot(run_id)
    source = snapshot.source+"\n"
    subject = snapshot.request.subject.model_copy(update={"content_hash":hashlib.sha256(source.encode()).hexdigest()})
    snapshot = snapshot.model_copy(update={"source":source,"request":snapshot.request.model_copy(update={"subject":subject})})
    snapshot = type(snapshot).model_validate(snapshot.model_dump(mode="json"))
    def forbidden(*_args,**_kwargs):
        """Detect any custom import after builtin source identity was already invalid."""
        pytest.fail("mismatched builtin bytes must not enter custom imports")
    monkeypatch.setattr(SourceLoader,"load",forbidden)
    output = validate_snapshot(snapshot,device="cpu")
    assert output.report.verdict == "invalid" and output.report.diagnostics[0].code == "builtin_compatibility"


@pytest.mark.parametrize("name",tuple(builtin_definitions()))
async def test_builtin_numerical_publication_accepts_exact_manifest_and_exposes_separate_copy_source(client,monkeypatch,name):
    """Real numerical fixture output enables one immutable builtin revision; HTTP source/copy reads retain distinct identities."""
    builtin_catalog.seed_blocks()
    definition = builtin_definitions()[name]
    with store.session() as db:
        run = next(row for row in db.scalars(select(Run)) if row.config["builtin_root"]["resource_id"] == str(definition.reference.resource_id))
        run_id = str(run.id)
    with monkeypatch.context() as patch:
        def forbidden(*_args,**_kwargs):
            """Builtin implementation digests must never enter the custom source loader."""
            pytest.fail("builtin root is not a custom source revision")
        patch.setattr(SourceLoader,"load",forbidden)
        snapshot,output = publish_fixture(run_id)
    assert output.report.verdict == "valid",output.report.diagnostics
    assert snapshot.builtin_root == definition.reference
    with store.session() as db:
        revision = catalog.require_revision(db,definition.reference)
        assert revision.source == readable_source(definition.reference)
        assert revision.acceptance_receipt_id == output.report.receipt.receipt_id
    admin = headers(["administration"])
    resource = definition.reference.resource_id
    root = f"/blocks/{resource}"
    detail = await client.get(root,headers=admin)
    assert detail.status_code == 200 and detail.json()["declaration"]["name"] == name
    source = await client.get(root+f"/revisions/{definition.reference.revision_id}",headers=admin)
    assert source.status_code == 200 and source.json()["source"] == snapshot.source
    copied = await client.get(root+f"/revisions/{definition.reference.revision_id}/copy-source",headers=admin)
    assert copied.status_code == 200 and "class CustomBlock" in copied.json()["source"]
    assert copied.json()["source"] != snapshot.source
    assert (await client.get(root+f"/revisions/{definition.reference.revision_id}/copy-source",headers=headers(["public"]))).status_code == 403
    assert (await client.put(root+"/draft",headers=admin,json={"expected_version":1,"source":"changed"})).status_code == 403
    revision_snapshot = snapshot.model_copy(update={"request":snapshot.request.model_copy(update={
        "subject":RevisionSubject(kind="revision",reference=definition.reference)})})
    copied_snapshot = type(snapshot).model_validate(revision_snapshot.model_dump(mode="json"))
    assert validate_snapshot(copied_snapshot,device="cpu").report.verdict == "valid"
    builtin_catalog.seed_blocks()
    with store.session() as db:
        assert len(list(db.scalars(select(Run)))) == 9


async def test_failed_or_pruned_builtin_bootstrap_never_creates_false_acceptance_or_replays_automatically(client):
    """An unsuccessful normal queue check remains visible/retryable and startup does not silently replace its identity."""
    builtin_catalog.seed_blocks()
    with store.session() as db:
        run = next(iter(db.scalars(select(Run))))
        run_id = str(run.id)
    admin = headers(["administration"])
    assert (await client.post(f"/runs/{run_id}/cancel",headers=admin)).status_code == 200
    assert (await client.delete(f"/runs/{run_id}",headers=admin)).status_code == 200
    builtin_catalog.seed_blocks()
    with store.session() as db:
        assert len(list(db.scalars(select(Run)))) == 8
        assert not list(db.scalars(select(BlockRevision)))
