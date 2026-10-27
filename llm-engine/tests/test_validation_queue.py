"""Durable validation HTTP/publication transactions and separate runnable real-worker acceptance."""

import asyncio
import threading
from uuid import UUID, uuid4

import jwt
import pytest
from httpx import ASGITransport, AsyncClient

import store
from api.runner import Runner
from api.runs import Registry
from api.validation_handler import ValidationHandler
from api.worker import SupervisedRun
from api import validation_jobs, validation_recovery
from llm.architecture.validation import validate_snapshot
from llm.architecture.validation_artifacts import publish_after_cleanup, read_publication
from llm.architecture.validation_records import ValidationSnapshot, subject_reference
from llm.architecture.builtins.catalog import copyable_source
from llm.architecture.spec import PendingTokenizer
from llm.architecture.wire import ValidationEvent, ValidationRun
from llm.runtime.supervisor import WorkerManager
from llm.runtime.supervisor import CleanupUncertain
from llm.runtime.artifacts import ArtifactStore
from llm.runtime.protocol import Completed
from llm.model.device import DeviceGate
from store import architecture_queries as catalog
from store.schema import Run, RunKind, RunStatus
from tests.test_architecture_core import SOURCE
from tests.test_architecture_validation import source_snapshot
from tests.test_architecture_validation import recipe_snapshot
from tests.test_architecture_visibility import SECRET, headers


@pytest.fixture
async def client(store_session,monkeypatch):
    """Use real signed ASGI routes and isolated storage without replacing authorization or running the queue implicitly."""
    import main

    monkeypatch.setenv("LLM_ENGINE_SERVICE_SECRET",SECRET)
    async with AsyncClient(transport=ASGITransport(app=main.app),base_url="http://engine") as value:
        yield value


async def submit(client, *, inspect: bool = False, source: str = SOURCE, admin: dict | None = None,
                 pending: bool = False, parameters: dict | None = None, binding=None):
    """Save a real draft and submit its exact immutable request, without executing source inside HTTP handlers."""
    admin = admin or headers(["administration"])
    created = await client.post("/blocks",headers=admin,json={"name":"Validation fixture","source":source})
    assert created.status_code == 201,created.text
    resource = created.json()["resource_id"]
    request = source_snapshot(source,inspect=inspect).request.model_dump(mode="json")
    request["subject"]["resource_id"] = resource
    if pending:
        request["tokenizer"] = PendingTokenizer(kind="pending",build_request_id=uuid4()).model_dump(mode="json")
    if parameters is not None:
        request["parameters"] = parameters
    if binding is not None:
        request["tokenizer"] = binding.model_dump(mode="json")
    response = await client.post(f"/blocks/{resource}/validations",headers=admin,json=request)
    assert response.status_code == 202,response.text
    ValidationRun.model_validate(response.json())
    return resource,request,response.json(),admin


def queued_snapshot(run_id: str) -> ValidationSnapshot:
    """Read the exact stored worker snapshot rather than reconstructing current catalog source."""
    with store.session() as db:
        return ValidationSnapshot.model_validate(db.get(Run,UUID(run_id)).config)


def publish_fixture(run_id: str):
    """Produce trusted numerical fixture output and exercise real publication/finalization without claiming worker containment."""
    snapshot = queued_snapshot(run_id)
    output = validate_snapshot(snapshot,device="cpu")
    publish_after_cleanup(snapshot,output)
    with store.session() as db:
        db.get(Run,snapshot.run_id).status = RunStatus.running
    Runner(Registry(),{})._finish(snapshot.run_id,RunStatus.done,result=output.report.model_dump(mode="json"),progress=1.0)
    return snapshot,output


async def test_submission_is_actor_idempotent_and_freezes_saved_bytes_before_edits(client,monkeypatch):
    """Identical actor/request retries retain one queued snapshot; conflicting arguments and stale subjects fail."""
    def forbidden(*_args,**_kwargs):
        """Detect any accidental source import during durable submission or catalog edits."""
        pytest.fail("submission must not execute source")
    monkeypatch.setattr("llm.architecture.source.SourceLoader.load",forbidden)
    resource,request,job,admin = await submit(client,inspect=True)
    repeated = await client.post(f"/blocks/{resource}/validations",headers=admin,json=request)
    assert repeated.status_code == 202 and repeated.json()["run_id"] == job["run_id"]
    changed = {**request,"mode":"full","profile":source_snapshot().request.profile.model_dump(mode="json")}
    conflict = await client.post(f"/blocks/{resource}/validations",headers=admin,json=changed)
    assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "request_conflict"
    assert (await client.put(f"/blocks/{resource}/draft",headers=admin,json={"expected_version":1,"source":"new invalid draft ["})).status_code == 200
    assert queued_snapshot(job["run_id"]).source == SOURCE
    stale = await client.post(f"/blocks/{resource}/validations",headers=admin,json={**request,"request_id":str(uuid4())})
    assert stale.status_code == 409
    assert (await client.get(f"/validations/{job['run_id']}",headers=admin)).status_code == 409
    assert "source" not in (await client.get(f"/runs/{job['run_id']}",headers=admin)).json()["config"]


async def test_retry_copies_pinned_source_and_receives_fresh_validation_ids_without_model_name(client):
    """Retrying cancelled validation preserves old bytes even when its catalog draft changes later."""
    resource,request,job,admin = await submit(client,inspect=True)
    run_id = job["run_id"]
    assert (await client.post(f"/runs/{run_id}/cancel",headers=admin)).status_code == 200
    await client.put(f"/blocks/{resource}/draft",headers=admin,json={"expected_version":1,"source":"different"})
    response = await client.post(f"/runs/{run_id}/retry",headers=admin)
    assert response.status_code == 202,response.text
    retried = ValidationRun.model_validate(response.json())
    assert str(retried.run_id) != run_id and str(retried.config.request_id) != request["request_id"]
    assert "model_name" not in response.text
    assert queued_snapshot(str(retried.run_id)).source == SOURCE


async def test_validation_retry_request_identity_replays_without_original_history_and_preserves_privacy(client):
    """Generic queue retry cannot duplicate a validation after a lost response, source pruning or administrator access loss."""
    _, _, job, admin = await submit(client, inspect=True)
    original = job["run_id"]
    assert (await client.post(f"/runs/{original}/cancel", headers=admin)).status_code == 200
    request_id = uuid4()
    path = f"/runs/{original}/retry?request_id={request_id}"
    first = await client.post(path, headers=admin)
    assert first.status_code == 202, first.text
    run = ValidationRun.model_validate(first.json())
    assert run.config.request_id == request_id
    assert str(run.run_id) != original
    assert (await client.delete(f"/runs/{original}", headers=admin)).status_code == 200
    replay = await client.post(path, headers=admin)
    assert replay.status_code == 202 and replay.json()["run_id"] == str(run.run_id)
    claims = jwt.decode(admin["Authorization"].split(" ", 1)[1], SECRET, audience="llm-engine", algorithms=["HS256"])
    claims["scope"] = ["public"]
    public = {"Authorization": "Bearer " + jwt.encode(claims, SECRET, algorithm="HS256")}
    assert (await client.post(path, headers=public)).status_code == 403
    conflict = await client.post(f"/runs/{run.run_id}/retry?request_id={request_id}", headers=admin)
    assert conflict.status_code == 409


async def test_published_source_reads_freeze_draft_bytes_and_require_admin(client, monkeypatch):
    """Source navigation returns the published draft's private import identity after edits, without rerunning inspection."""
    resource, _, job, admin = await submit(client, inspect=True)
    snapshot, _ = publish_fixture(job["run_id"])
    reference = subject_reference(snapshot)
    path = f"/validations/{job['run_id']}/sources/{reference.revision_id}"
    await client.put(f"/blocks/{resource}/draft", headers=admin, json={"expected_version":1,"source":"changed draft"})

    def forbidden(*args, **kwargs):
        """Refuse construction/import/tensor reads during frozen-source presentation."""
        pytest.fail("source read attempted execution")

    monkeypatch.setattr("llm.architecture.source.SourceLoader.load", forbidden)
    monkeypatch.setattr("llm.architecture.validation.validate_snapshot", forbidden)
    monkeypatch.setattr("torch.load", forbidden)
    response = await client.get(path, headers=admin)
    assert response.status_code == 200, response.text
    assert response.json()["source"] == SOURCE
    assert response.json()["content_hash"] == reference.content_hash
    assert response.json()["definition_id"] is None
    assert (await client.get(path, headers=headers(["public"]))).status_code == 403
    assert (await client.get(f"/validations/{job['run_id']}/sources/{uuid4()}", headers=admin)).status_code == 404


@pytest.mark.parametrize("inspect",[True,False])
async def test_published_report_graph_and_acceptance_follow_terminal_evidence_transaction(client,inspect):
    """Actual data/HTTP adapters require complete publication before evidence and preserve acceptance after history deletion."""
    resource,request,job,admin = await submit(client,inspect=inspect)
    snapshot,output = publish_fixture(job["run_id"])
    response = await client.get(f"/validations/{snapshot.run_id}",headers=admin)
    assert response.status_code == 200 and response.json()["verdict"] == ("inspection_only" if inspect else "valid")
    acceptance = {"expected_draft_version":1,"validation_run_id":job["run_id"],
                  "expected_content_hash":request["subject"]["content_hash"],
                  "expected_dependency_digest":catalog.dependency_digest({})}
    if inspect:
        assert (await client.post(f"/blocks/{resource}/revisions",headers=admin,json=acceptance)).status_code == 409
        return
    receipt = output.report.receipt
    query = {"identity_hash":receipt.identity_hash,"view":"operations","mode":"eval","signature":"full_prefix","limit":2}
    graph = await client.get(f"/validations/{snapshot.run_id}/graph",headers=admin,params=query)
    assert graph.status_code == 200 and graph.json()["graph_hash"] == receipt.graph_manifest_hash
    stale = await client.get(f"/validations/{snapshot.run_id}/graph",headers=admin,params={**query,"identity_hash":"b"*64})
    assert stale.status_code == 409
    accepted = await client.post(f"/blocks/{resource}/revisions",headers=admin,json=acceptance)
    assert accepted.status_code == 201,accepted.text
    same = await client.post(f"/blocks/{resource}/revisions",headers=admin,json=acceptance)
    assert same.status_code == 201 and same.json() == accepted.json()
    assert (await client.delete(f"/runs/{snapshot.run_id}",headers=admin)).status_code == 200
    assert (await client.get(f"/validations/{snapshot.run_id}",headers=admin)).status_code == 200
    assert (await client.get(f"/blocks/{resource}/revisions/{accepted.json()['reference']['revision_id']}",headers=admin)).status_code == 200


async def test_invalid_and_pending_results_are_not_acceptance_or_graph_authority(client):
    """Completed-invalid and provisional metadata stay distinct from worker failure and full numerical evidence."""
    for pending in (False,True):
        source = SOURCE.replace("super().__init__()","raise RuntimeError('constructor must not authorize')")
        resource,request,job,admin = await submit(client,source=source,pending=pending)
        snapshot,output = publish_fixture(job["run_id"])
        assert output.report.verdict == ("provisional" if pending else "invalid"),output.report.diagnostics
        with store.session() as db:
            assert db.get(Run,snapshot.run_id).status == RunStatus.done
            assert catalog.evidence_for_job(db,snapshot.run_id) is None
        graph = await client.get(f"/validations/{snapshot.run_id}/graph",headers=admin,params={
            "identity_hash":output.report.receipt.identity_hash if pending else "a"*64,
            "view":"operations","mode":"eval","signature":"full_prefix"})
        assert graph.status_code == 409
        accepted = await client.post(f"/blocks/{resource}/revisions",headers=admin,json={
            "expected_draft_version":1,"validation_run_id":job["run_id"],"expected_content_hash":request["subject"]["content_hash"],
            "expected_dependency_digest":catalog.dependency_digest({})})
        assert accepted.status_code == 409


async def test_public_scope_cannot_read_cancel_retry_delete_or_reorder_private_validations(client):
    """All private validation mutations enforce trusted scope before revealing or changing durable state."""
    _,_,job,_ = await submit(client,inspect=True)
    public = headers(["public"])
    for method,path in (("get",f"/validations/{job['run_id']}"),("post",f"/runs/{job['run_id']}/cancel"),
                        ("post",f"/runs/{job['run_id']}/retry"),("delete",f"/runs/{job['run_id']}"),
                        ("patch",f"/runs/{job['run_id']}")):
        response = await getattr(client,method)(path,headers=public)
        assert response.status_code == 403,response.text


async def test_missing_publication_rolls_back_completion_and_cancelled_publication_cannot_authorize(client):
    """A missing artifact/evidence transaction cannot leave a done row; cancellation wins before evidence retention."""
    _,_,job,_ = await submit(client)
    run_id = UUID(job["run_id"])
    with store.session() as db:
        db.get(Run,run_id).status = RunStatus.running
    runner = Runner(Registry(),{})
    with pytest.raises(FileNotFoundError):
        runner._finish(run_id,RunStatus.done,result={"verdict":"valid"})
    with store.session() as db:
        assert db.get(Run,run_id).status == RunStatus.running
    snapshot = queued_snapshot(str(run_id))
    output = validate_snapshot(snapshot,device="cpu")
    publish_after_cleanup(snapshot,output)
    with store.session() as db:
        db.get(Run,run_id).status = RunStatus.cancelling
    assert runner._finish(run_id,RunStatus.done,result=output.report.model_dump(mode="json")) == RunStatus.cancelled
    with store.session() as db:
        assert catalog.evidence_for_job(db,run_id) is None


async def test_restart_adopts_only_complete_running_publications_and_refuses_tampered_bytes(client):
    """Startup adoption completes an interrupted registration only after a complete matching parent publication."""
    _,_,job,_ = await submit(client)
    snapshot = queued_snapshot(job["run_id"])
    output = validate_snapshot(snapshot,device="cpu")
    publication = publish_after_cleanup(snapshot,output)
    with store.session() as db:
        db.get(Run,snapshot.run_id).status = RunStatus.running
    validation_recovery.reconcile()
    with store.session() as db:
        assert db.get(Run,snapshot.run_id).status == RunStatus.done
        assert catalog.evidence_for_job(db,snapshot.run_id).receipt_id == publication.output.report.receipt.receipt_id
    from llm import storage
    (storage.validation_dir(snapshot.run_id)/"validation.json").write_text("{}")
    with pytest.raises(ValueError,match="corrupt"):
        read_publication(snapshot.run_id)


async def test_restart_snapshot_mismatch_rolls_back_success_and_continues_other_jobs(client):
    """A complete publication for changed queue bytes cannot authorize evidence or prevent recovery of an independent job."""
    snapshots = []
    for _ in range(2):
        _,_,job,_ = await submit(client,inspect=True)
        snapshot = queued_snapshot(job["run_id"])
        publish_after_cleanup(snapshot,validate_snapshot(snapshot,device="cpu"))
        with store.session() as db:
            run = db.get(Run,snapshot.run_id)
            run.status = RunStatus.running
            if not snapshots:
                run.config = snapshot.model_copy(update={"request":snapshot.request.model_copy(update={"request_id":uuid4()})}).model_dump(mode="json")
        snapshots.append(snapshot)
    validation_recovery.reconcile()
    with store.session() as db:
        assert db.get(Run,snapshots[0].run_id).status == RunStatus.running
        assert db.get(Run,snapshots[1].run_id).status == RunStatus.done
        assert catalog.evidence_for_job(db,snapshots[0].run_id) is None


async def test_copied_builtin_validation_pins_implementation_before_import_and_acceptance(client,monkeypatch):
    """Copied public-builtin code cannot reuse evidence from a different installed implementation snapshot."""
    source = copyable_source("normalization").replace("from llm.architecture.builtins", "from llm . architecture . builtins")
    resource,request,job,admin = await submit(client,source=source,parameters={"width":8})
    snapshot = queued_snapshot(job["run_id"])
    assert snapshot.builtins
    with pytest.raises(ValueError,match="implementation pins"):
        ValidationSnapshot.model_validate({**snapshot.model_dump(mode="json"),"builtins":[]})
    first = snapshot.builtins[0].model_copy(update={"content_hash":"b"*64})
    stale = snapshot.model_copy(update={"builtins":(first,*snapshot.builtins[1:])})
    with monkeypatch.context() as patch:
        def forbidden(*_args,**_kwargs):
            """Ensure stale builtin pins fail before custom Python can import anything."""
            pytest.fail("source must not import with stale builtin pins")
        patch.setattr("llm.architecture.source.SourceLoader.load",forbidden)
        rejected = validate_snapshot(stale,device="cpu")
    assert rejected.report.diagnostics[0].code == "builtin_compatibility"
    _,output = publish_fixture(job["run_id"])
    assert output.report.verdict == "valid",output.report.diagnostics
    accepted = await client.post(f"/blocks/{resource}/revisions",headers=admin,json={
        "expected_draft_version":1,"validation_run_id":job["run_id"],"expected_content_hash":request["subject"]["content_hash"],
        "expected_dependency_digest":output.report.receipt.identity.dependency_digest})
    assert accepted.status_code == 201,accepted.text


async def test_cancelled_validation_discards_completed_report_from_durable_result(client):
    """Cancellation cannot expose a numerically valid report through a terminal cancelled run."""
    _,_,job,admin = await submit(client)
    snapshot = queued_snapshot(job["run_id"])
    output = validate_snapshot(snapshot,device="cpu")
    with store.session() as db:
        db.get(Run,snapshot.run_id).status = RunStatus.running
    Runner(Registry(),{})._finish(snapshot.run_id,RunStatus.cancelled,result=output.report.model_dump(mode="json"))
    with store.session() as db:
        run = db.get(Run,snapshot.run_id)
        assert run.status == RunStatus.cancelled and run.result is None
        assert catalog.evidence_for_job(db,snapshot.run_id) is None
    assert (await client.get(f"/validations/{snapshot.run_id}",headers=admin)).status_code == 409


@pytest.mark.parametrize("changes",[{"limit":257},{"limit":0},{"scope":""},{"cursor":"x"*2049}])
async def test_graph_http_bounds_reject_before_reading_artifacts(client,changes,monkeypatch):
    """Malformed graph pages cannot trigger publication reads or model capture."""
    _,_,job,admin = await submit(client,inspect=True)
    def forbidden(*_args,**_kwargs):
        """Detect artifact reads attempted before HTTP boundary validation."""
        pytest.fail("invalid graph query must not read artifacts")
    monkeypatch.setattr(validation_jobs,"published_result",forbidden)
    query = {"identity_hash":"a"*64,"view":"operations","mode":"eval","signature":"full_prefix",**changes}
    response = await client.get(f"/validations/{job['run_id']}/graph",headers=admin,params=query)
    assert response.status_code == 422,response.text


async def test_archived_root_refuses_validation_and_pruned_submission_requires_new_request(client):
    """New admission checks current archive state, while idempotent history cannot silently recreate a deleted job."""
    resource,request,job,admin = await submit(client,inspect=True)
    assert (await client.post(f"/runs/{job['run_id']}/cancel",headers=admin)).status_code == 200
    assert (await client.delete(f"/runs/{job['run_id']}",headers=admin)).status_code == 200
    replay = await client.post(f"/blocks/{resource}/validations",headers=admin,json=request)
    assert replay.status_code == 404 and replay.json()["error"]["code"] == "operation_pruned"
    assert (await client.patch(f"/blocks/{resource}",headers=admin,json={"expected_metadata_version":1,"archived":True})).status_code == 200
    archived = await client.post(f"/blocks/{resource}/validations",headers=admin,json={**request,"request_id":str(uuid4())})
    assert archived.status_code == 409 and archived.json()["error"]["code"] == "archived"


async def test_final_tokenizer_facts_come_from_owned_bytes_and_mismatches_roll_back_queue_admission(client):
    """User-supplied vocabulary/hash/reserved IDs or derived provenance cannot fabricate finalized tokenizer authority."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from llm import storage
    from llm.architecture.tokenizer_binding import binding_from_bytes
    from sqlalchemy import select

    tokenizer_id = uuid4()
    raw = Tokenizer(WordLevel(vocab={"<|endoftext|>":0,"one":1,"two":2},unk_token="<|endoftext|>")).to_str().encode()
    directory = storage.tokenizer_build_dir(str(tokenizer_id))
    directory.mkdir(parents=True)
    (directory/"tokenizer.json").write_bytes(raw)
    (directory/"build.json").write_text('{"vocab_size":3}')
    binding = binding_from_bytes(tokenizer_id,raw,("pad_id",))
    resource,request,job,admin = await submit(client,binding=binding)
    assert queued_snapshot(job["run_id"]).request.tokenizer == binding
    read = await client.get(f"/tokenizers/{tokenizer_id}/architecture-binding",headers=admin)
    assert read.status_code == 200 and read.json() == binding_from_bytes(tokenizer_id,raw).model_dump(mode="json")
    reserved = await client.get(f"/tokenizers/{tokenizer_id}/architecture-binding?reserved=pad_id",headers=admin)
    assert reserved.status_code == 200 and reserved.json() == binding.model_dump(mode="json")
    assert (await client.get(f"/tokenizers/{tokenizer_id}/architecture-binding",headers=headers(["public"]))).status_code == 403
    for changes in ({"fingerprint":"f"*64},{"vocabulary_size":4},{"reserved_token_ids":{"pad_id":1}},
                    {"derived_from":"a"*64},{"tokenizer_id":uuid4()}):
        wrong = binding.model_copy(update=changes)
        response = await client.post(f"/blocks/{resource}/validations",headers=admin,json={**request,
            "request_id":str(uuid4()),"tokenizer":wrong.model_dump(mode="json")})
        assert response.status_code == 422 and response.json()["error"]["code"] == "tokenizer_binding",response.text
    with store.session() as db:
        assert len(list(db.scalars(select(Run)))) == 1
    malformed = await client.get(f"/tokenizers/{tokenizer_id}/architecture-binding?reserved=not.valid",headers=admin)
    assert malformed.status_code == 422 and malformed.json()["error"]["code"] == "invalid_request"
    assert b'"two":2' in raw
    (directory/"tokenizer.json").write_bytes(raw.replace(b'"two":2',b'"two":4'))
    holes = await client.get(f"/tokenizers/{tokenizer_id}/architecture-binding",headers=admin)
    assert holes.status_code == 422 and holes.json()["error"]["code"] == "tokenizer_binding"


def test_pending_causal_recipe_never_constructs_placeholder_vocabulary(monkeypatch):
    """Pending tokenizer metadata remains provisional with skipped numerical stages and no graph or acceptance authority."""
    snapshot = recipe_snapshot(tokenizer=PendingTokenizer(kind="pending",build_request_id=uuid4()))
    def forbidden(*_args,**_kwargs):
        """Refuse any construction before the authoritative tokenizer has been finalized."""
        pytest.fail("pending tokenizer must not construct a model")
    monkeypatch.setattr("llm.architecture.validation.constructed_model",forbidden)
    output = validate_snapshot(snapshot,device="cpu")
    assert output.report.verdict == "provisional",output.report.diagnostics
    assert not output.graph_blobs and output.report.receipt.graph_manifest_hash is None
    assert {stage.outcome for stage in output.report.stages if stage.stage in {"capture","numerical"}} == {"skipped"}
    with pytest.raises(ValueError):
        output.report.receipt.require_current(output.report.receipt.identity)


@pytest.mark.parametrize("cleanup_fails",[False,True])
def test_handler_publication_follows_cleanup_boundary_with_no_evidence_on_cleanup_failure(tmp_path,monkeypatch,cleanup_fails):
    """A worker seam fixture proves publication ordering only; the separate supervised test proves real containment."""
    from api import validation_handler
    from llm import storage

    snapshot = source_snapshot(inspect=True)
    output = validate_snapshot(snapshot,device="cpu")
    order = []
    class Session:
        """Provide real bounded grants while simulating only the process cleanup outcome."""
        def __init__(self,policy):
            """Own a fixture artifact directory under the supplied production resource policy."""
            self.artifacts = ArtifactStore(tmp_path/"grants",policy.resources)
        def __enter__(self):
            """Mark the fixture worker as running before its immutable command is sent."""
            order.append("entered")
            return self
        def run(self,payload,**_kwargs):
            """Validate input grant bytes and return an already computed trusted fixture report."""
            assert ValidationSnapshot.model_validate_json(self.artifacts.read_bytes(payload.snapshot)) == snapshot
            result = self.artifacts.write_bytes(output.model_dump_json().encode(),"validation_result_v1")
            return Completed(type="completed",outcome="ok",result=result)
        def __exit__(self,*_args):
            """Report cleanup success or explicit uncertainty before the handler can publish anything."""
            order.append("cleanup")
            if cleanup_fails:
                raise CleanupUncertain("fixture cleanup uncertainty")
    class Workers:
        """Expose the existing worker session seam without pretending this fixture launches a process."""
        def session(self,lease,policy,_backend,**kwargs):
            """Require declaration-only CPU admission for this inspection command."""
            assert lease is None and kwargs["metadata_only"]
            return Session(policy)
    original = validation_handler.publish_after_cleanup
    def publish(*args):
        """Assert the ordering invariant at the actual durable artifact publication boundary."""
        assert order[-1] == "cleanup"
        order.append("published")
        return original(*args)
    monkeypatch.setattr(validation_handler,"publish_after_cleanup",publish)
    with DeviceGate().admit() as lease:
        context = SupervisedRun(job_id=snapshot.run_id,config=snapshot.model_dump(mode="json"),actor=None,
            progress=lambda _:None,cancel=threading.Event(),lease=lease,workers=Workers())
        if cleanup_fails:
            with pytest.raises(CleanupUncertain):
                ValidationHandler().run_supervised(context)
            assert not storage.validation_dir(snapshot.run_id).exists()
        else:
            assert ValidationHandler().run_supervised(context)["verdict"] == "inspection_only"
            assert order == ["entered","cleanup","published"]


async def test_supervised_http_validation_acceptance_uses_actual_worker_cleanup(client,tmp_path):
    """Run real HTTP submission through the production queue/worker/publication/acceptance seam when processes are available."""
    resource,request,job,admin = await submit(client)
    registry = Registry()
    registry.bind_loop(asyncio.get_running_loop())
    with WorkerManager(tmp_path/"runtime") as manager:
        runner = Runner(registry,{RunKind.model_validation:ValidationHandler()})
        runner.attach_workers(manager)
        await asyncio.wait_for(runner._drain_once(),60)
    report = await client.get(f"/validations/{job['run_id']}",headers=admin)
    assert report.status_code == 200 and report.json()["verdict"] == "valid",report.text
    terminal = registry.get(job["run_id"]).history[-1]
    assert ValidationEvent.model_validate(terminal.data).type == "done"
    accepted = await client.post(f"/blocks/{resource}/revisions",headers=admin,json={
        "expected_draft_version":1,"validation_run_id":job["run_id"],"expected_content_hash":request["subject"]["content_hash"],
        "expected_dependency_digest":catalog.dependency_digest({})})
    assert accepted.status_code == 201,accepted.text
