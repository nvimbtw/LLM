"""In-process real route/storage catalog checks without numerical workers or TCP fixtures."""

from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

import store
from llm.architecture.spec import MAX_SOURCE_BYTES
from llm.architecture.wire import BlockCreate, BlockDraftSave
from store import architecture_queries as q
from store.schema import Run
from tests.test_architecture_store import acceptance, block_declaration, evidence
from tests.test_architecture_visibility import SECRET, headers


@pytest.fixture
async def client(store_session,monkeypatch):
    """Exercise actual signed route dependencies and isolated SQLite without starting queue/process owners."""
    import main

    monkeypatch.setenv("LLM_ENGINE_SERVICE_SECRET",SECRET)
    async with AsyncClient(transport=ASGITransport(app=main.app),base_url="http://engine") as caller:
        yield caller


@pytest.mark.parametrize("path,field",[("blocks","source"),("architectures","yaml")])
async def test_invalid_draft_crud_never_imports_source_and_preserves_versions(client,monkeypatch,path,field):
    """Saved bytes are editable independently of validation; stale saves/metadata cannot overwrite current data."""
    def forbidden(*_args,**_kwargs):
        """Fail if source inspection or editable YAML interpretation leaks into a catalog operation."""
        pytest.fail("catalog operations must not execute or parse draft source")
    monkeypatch.setattr("llm.architecture.source.SourceLoader.load",forbidden)
    monkeypatch.setattr("llm.architecture.parse.parse_recipe",forbidden)
    admin = headers(["administration"])
    created = await client.post(f"/{path}",headers=admin,json={"name":"Invalid draft",field:"def broken("})
    assert created.status_code == 201,created.text
    identity = created.json()["resource_id"]
    root = f"/{path}/{identity}"
    detail = (await client.get(root,headers=admin)).json()
    assert detail["revision_count"] == 0 and detail["summary"]["accepted_revision"] is None
    assert field not in detail["summary"]
    draft = (await client.get(root+"/draft",headers=admin)).json()
    assert draft[field] == "def broken(" and draft["draft_version"] == 1
    saved = await client.put(root+"/draft",headers=admin,json={"expected_version":1,field:"still invalid ["})
    assert saved.status_code == 200 and saved.json()["draft_version"] == 2
    stale = await client.put(root+"/draft",headers=admin,json={"expected_version":1,field:"overwrite"})
    assert stale.status_code == 409 and stale.json()["current_version"] == 2
    assert (await client.get(root+"/draft",headers=admin)).json()[field] == "still invalid ["
    changed = await client.patch(root,headers=admin,json={"expected_metadata_version":1,"description":"Updated","archived":True})
    assert changed.status_code == 200 and changed.json()["summary"]["metadata_version"] == 2
    assert (await client.patch(root,headers=admin,json={"expected_metadata_version":1,"name":"Late"})).status_code == 409
    assert not (await client.get(f"/{path}",headers=admin)).json()["items"]
    assert len((await client.get(f"/{path}?archived=true",headers=admin)).json()["items"]) == 1


async def test_catalog_pages_are_bounded_and_cursors_reject_filter_family_replay(client):
    """Keyset pages cannot mix block/architecture/filter identities or silently truncate continuation."""
    admin = headers(["administration"])
    for name in ("Alpha","Beta","Gamma"):
        assert (await client.post("/blocks",headers=admin,json={"name":name,"source":"invalid source"})).status_code == 201
    page = (await client.get("/blocks?limit=2",headers=admin)).json()
    assert len(page["items"]) == 2 and page["next_cursor"]
    cursor = page["next_cursor"]
    second = (await client.get("/blocks",headers=admin,params={"limit":2,"cursor":cursor})).json()
    assert len(second["items"]) == 1 and second["next_cursor"] is None
    assert not {row["resource_id"] for row in page["items"]} & {row["resource_id"] for row in second["items"]}
    for path,params in (("/architectures",{"cursor":cursor}),("/blocks",{"cursor":cursor,"search":"Alpha"})):
        response = await client.get(path,headers=admin,params=params)
        assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_cursor"
    assert (await client.get("/blocks?limit=101",headers=admin)).status_code == 422


@pytest.mark.parametrize("scope",[None,[],["public"],"administration",["administration","public"]])
async def test_catalog_and_schema_require_exact_signed_administrative_scope(client,scope):
    """Browser visibility parameters and malformed scope carriers cannot authorize administrative data."""
    for path in ("/blocks","/architectures","/architectures/schema",f"/blocks/{uuid4()}/draft"):
        response = await client.get(path+"?visibility=administration",headers=headers(scope))
        assert response.status_code == 403,response.text
    assert (await client.post("/blocks",headers=headers(scope),json={"name":"No","source":"secret"})).status_code == 403


async def test_read_only_builtin_missing_identity_schema_precedence_and_utf8_budget(client):
    """Literal schema precedes UUID routes, and body/value limits/read-only/not-found refusals keep their statuses."""
    admin = headers(["administration"])
    schema = await client.get("/architectures/schema",headers=admin)
    assert schema.status_code == 200 and schema.json()["recipe"]["title"] == "Recipe"
    with store.session() as db:
        row = q.create_block(db,uuid4(),BlockCreate(name="Builtin",source="stored"),builtin=True)
        identity = row.id
    assert (await client.put(f"/blocks/{identity}/draft",headers=admin,json={"expected_version":1,"source":"changed"})).status_code == 403
    assert (await client.patch(f"/blocks/{identity}",headers=admin,json={"expected_metadata_version":1,"archived":True})).status_code == 403
    assert (await client.get(f"/blocks/{uuid4()}",headers=admin)).status_code == 404
    oversized = await client.post("/blocks",headers=admin,json={"name":"Bytes","source":"é"*(MAX_SOURCE_BYTES//2+1)})
    assert oversized.status_code == 413 and oversized.json()["error"]["code"] == "source_byte_limit"
    invalid = await client.post("/architectures",headers=admin,json={"name":"Bad","yaml":"draft","unknown":1})
    assert invalid.status_code == 422 and invalid.json()["error"]["code"] == "invalid_request"


async def test_immutable_revision_reads_preserve_source_and_evidence_after_draft_edit_and_job_pruning(client):
    """Stored fixture acceptance reads remain available independently of changed drafts and deleted validation jobs."""
    admin = headers(["administration"])
    with store.session() as db:
        row = q.create_block(db,uuid4(),BlockCreate(name="Accepted fixture",source="old immutable bytes"))
        run,receipt = evidence(db,row,q.dependency_digest({}),{})
        revision = q.accept_block(db,row.id,acceptance(row,run,receipt),block_declaration())
        resource_id,revision_id,run_id = row.id,revision.id,run.id
        q.save_block_draft(db,row.id,BlockDraftSave(expected_version=1,source="new invalid draft ["))
    with store.session() as db:
        db.delete(db.get(Run,run_id))
    root = f"/blocks/{resource_id}"
    page = (await client.get(root+"/revisions",headers=admin)).json()
    assert page["items"][0]["reference"]["revision_id"] == str(revision_id)
    assert page["items"][0]["acceptance_identity_hash"] == receipt.identity_hash
    saved = (await client.get(root+f"/revisions/{revision_id}",headers=admin)).json()
    assert saved["source"] == "old immutable bytes"
    assert (await client.get(root+"/draft",headers=admin)).json()["source"] == "new invalid draft ["
    assert (await client.get(f"/blocks/{uuid4()}/revisions/{revision_id}",headers=admin)).status_code == 404
    detail = (await client.get(root,headers=admin)).json()
    assert detail["revision_count"] == 1 and detail["declaration"] is not None
