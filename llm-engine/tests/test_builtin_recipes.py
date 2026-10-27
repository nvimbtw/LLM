"""Real tokenizer/recipe bootstrap publication through parent adapters; live process evidence is a separate runnable check."""

import asyncio
import hashlib
from uuid import uuid4

import pytest
from sqlalchemy import select

import store
from api import builtin_catalog, builtin_tokenizer, validation_jobs
from api.runner import Runner
from api.runs import Registry
from api.validation_handler import ValidationHandler
from llm import storage
from llm.architecture.parse import parse_recipe
from llm.architecture.resolve import parameter_values
from llm.architecture.source import SourceLoader
from llm.architecture.spec import PendingTokenizer, canonical_digest
from llm.architecture.wire import ArchitectureCreate, EnqueueEvidence
from llm.corpus import baseline, bpe, special, vocab
from llm.runtime.supervisor import WorkerManager
from store import architecture_queries as catalog, queries
from store.schema import Architecture, ArchitectureRevision, BlockRevision, Run, RunArchitecturePin, RunKind, RunStatus, Tokenizer
from tests.test_architecture_visibility import headers
from tests.test_validation_queue import client, publish_fixture


@pytest.fixture(autouse=True)
def builtin_data(tmp_path,monkeypatch):
    """Give each bootstrap/artifact failure test its own complete data root without touching the shared tiny model."""
    monkeypatch.setenv("DATA_DIR",str(tmp_path/"builtin-data"))


def finish_blocks() -> None:
    """Publish trusted numerical fixtures for all seeded block jobs; this helper does not claim child-process containment."""
    with store.session() as db:
        ids = [str(row.id) for row in db.scalars(select(Run)) if row.config.get("builtin_root") is not None]
    for run_id in ids:
        _,output = publish_fixture(run_id)
        assert output.report.verdict == "valid",output.report.diagnostics


def test_byte_alphabet_is_real_deterministic_and_covers_unicode_and_reserved_ids():
    """The concrete example tokenizer round-trips every byte, preserves actual reserved IDs and contains no learned merges."""
    first,second = baseline.byte_alphabet(),baseline.byte_alphabet()
    assert first.to_str() == second.to_str()
    assert first.get_vocab_size() == len(special.token_list())+256
    assert bpe.merge_count(first) == 0
    for text in ("Hello, ž🙂", "\x00\n\t"+"".join(chr(value) for value in range(256)), "<|user|>hello<|/user|>"):
        assert bpe.decode(first,bpe.encode(first,text)) == text
    assert [first.token_to_id(marker) for marker in special.token_list()] == list(range(len(special.token_list())))


def test_baseline_post_rename_index_failure_retains_exact_orphan_and_preserves_later_display_edits(store_session,monkeypatch):
    """A durable tokenizer orphan can be indexed on restart without replacing bytes or resetting existing display metadata."""
    tokenizer_id,raw = builtin_tokenizer.baseline_bytes()
    with monkeypatch.context() as patch:
        def failed_index(*_args,**_kwargs):
            """Fail registration only after artifact publication has succeeded."""
            raise RuntimeError("index unavailable")
        patch.setattr(queries,"upsert_tokenizer",failed_index)
        with pytest.raises(RuntimeError,match="index unavailable"):
            builtin_tokenizer.ensure_baseline()
    assert storage.tokenizer_file(storage.tokenizer_build_dir(str(tokenizer_id))).read_bytes() == raw
    with store.session() as db:
        assert db.get(Tokenizer,tokenizer_id) is None
    binding = builtin_tokenizer.ensure_baseline()
    assert binding.fingerprint == hashlib.sha256(raw).hexdigest()
    assert validation_jobs.read_tokenizer_binding(tokenizer_id) == binding
    with store.session() as db:
        row = db.get(Tokenizer,tokenizer_id)
        row.name,row.description = "My display name","My description"
    assert builtin_tokenizer.ensure_baseline() == binding
    with store.session() as db:
        assert db.get(Tokenizer,tokenizer_id).name == "My display name"
        assert db.get(Tokenizer,tokenizer_id).description == "My description"
    assert vocab.load(str(tokenizer_id)).get_vocab_size() == binding.vocabulary_size


def test_interrupted_baseline_publication_never_creates_index_or_replaces_conflicting_bytes(store_session,monkeypatch):
    """Pre-rename write failure leaves no complete build; an unrelated same-UUID artifact is refused and preserved."""
    tokenizer_id,raw = builtin_tokenizer.baseline_bytes()
    original = builtin_tokenizer.write_bytes
    def failed_write(path,data):
        """Interrupt after the tokenizer bytes, before the build marker can be published."""
        if path.name == "build.json":
            raise OSError("interrupted marker")
        original(path,data)
    with monkeypatch.context() as patch:
        patch.setattr(builtin_tokenizer,"write_bytes",failed_write)
        with pytest.raises(OSError,match="interrupted marker"):
            builtin_tokenizer.ensure_baseline()
    assert not storage.tokenizer_build_dir(str(tokenizer_id)).exists()
    with store.session() as db:
        assert db.get(Tokenizer,tokenizer_id) is None
    directory = storage.tokenizer_build_dir(str(tokenizer_id))
    directory.mkdir()
    (directory/"tokenizer.json").write_bytes(b"conflicting original")
    (directory/"build.json").write_text("{}")
    with pytest.raises(catalog.CatalogError,match="conflicting"):
        builtin_tokenizer.ensure_baseline()
    assert (directory/"tokenizer.json").read_bytes() == b"conflicting original"


def test_empty_install_seeds_only_real_artifacts_and_waits_for_accepted_blocks(store_session,monkeypatch):
    """Data-only startup creates four immutable-pin recipe drafts and a real tokenizer, without model construction or invented acceptance."""
    def forbidden(*_args,**_kwargs):
        """Detect any numerical model/source execution in startup catalog setup."""
        pytest.fail("startup must not construct or import model source")
    monkeypatch.setattr("torch.nn.Module.__init__",forbidden)
    monkeypatch.setattr(SourceLoader,"load",forbidden)
    builtin_catalog.seed_blocks()
    builtin_catalog.seed_recipes()
    builtin_catalog.seed_recipes()
    with store.session() as db:
        assert len(list(db.scalars(select(Architecture)))) == 4
        assert len(list(db.scalars(select(Run)))) == 9
        assert not list(db.scalars(select(BlockRevision)))
        assert not list(db.scalars(select(ArchitectureRevision)))
        assert all(row.config["request"]["tokenizer"] is None for row in db.scalars(select(Run)))
        tokenizer = db.scalar(select(Tokenizer))
        assert tokenizer.implicit and tokenizer.merges == 0
        binding = validation_jobs.read_tokenizer_binding(tokenizer.id)
    assert binding.vocabulary_size == len(special.token_list())+256


def test_supplied_recipe_identity_cannot_schedule_a_custom_resource(store_session):
    """A foreign custom row at a reserved supplied UUID must fail before its source can be interpreted or queued."""
    builtin_tokenizer.ensure_baseline()
    with store.session() as db:
        catalog.create_architecture(db,uuid4(),ArchitectureCreate(name="Foreign",yaml="invalid source"),
            resource_id=builtin_catalog.recipe_resource_id("standard"))
    with store.session() as db:
        with pytest.raises(catalog.CatalogError,match="custom resource"):
            builtin_catalog.queue_ready_recipes(db)
        assert not list(db.scalars(select(Run)))


async def test_all_builtin_recipes_accept_concrete_examples_and_pending_training_still_requires_final_preflight(client):
    """Actual numerical fixtures accept all provided recipes; their example receipt cannot authorize pending training bindings."""
    builtin_catalog.seed_blocks()
    builtin_catalog.seed_recipes()
    finish_blocks()
    with store.session() as db:
        ids = [str(row.id) for row in db.scalars(select(Run)) if row.config.get("recipe") is not None]
    assert len(ids) == 4
    for run_id in ids:
        snapshot,output = publish_fixture(run_id)
        assert output.report.verdict == "valid",output.report.diagnostics
        assert snapshot.request.tokenizer == validation_jobs.read_tokenizer_binding(snapshot.request.tokenizer.tokenizer_id)
    admin = headers(["administration"])
    for name in builtin_catalog.RECIPE_NAMES:
        resource_id = builtin_catalog.recipe_resource_id(name)
        root = f"/architectures/{resource_id}"
        detail = await client.get(root,headers=admin)
        assert detail.status_code == 200 and detail.json()["revision_count"] == 1
        assert (await client.put(root+"/draft",headers=admin,json={"expected_version":1,"yaml":"changed"})).status_code == 403
        with store.session() as db:
            revision = catalog.revision_rows(db,"architecture",resource_id)[0]
            receipt = catalog._receipt(db,revision.acceptance_receipt_id)
            parameters = parameter_values(parse_recipe(revision.yaml).parameters,{})
            pending = PendingTokenizer(kind="pending",build_request_id=uuid4())
            selection = EnqueueEvidence(architecture=catalog.revision_reference(revision),dependency_digest=revision.dependency_digest,
                parameter_digest=canonical_digest(parameters),acceptance_receipt_id=receipt.receipt_id,
                tokenizer=pending,exact_receipt_id=None,final_preflight_required=True)
            run = catalog.enqueue_pinned_training(db,actor_id=uuid4(),label="Future training",config={},
                evidence=selection,parameters=parameters)
            assert db.get(RunArchitecturePin,run.id).evidence["final_preflight_required"]
            with pytest.raises(ValueError):
                EnqueueEvidence.model_validate({
                    **selection.model_dump(mode="json"),"exact_receipt_id":str(receipt.receipt_id),"final_preflight_required":False})
    builtin_catalog.seed_blocks()
    builtin_catalog.seed_recipes()
    with store.session() as db:
        assert len(list(db.scalars(select(ArchitectureRevision)))) == 4
        assert len([row for row in db.scalars(select(Run)) if row.kind == RunKind.model_validation]) == 13
        resource_id = builtin_catalog.recipe_resource_id("standard")
        row = db.get(Architecture,resource_id)
        original = ArchitectureCreate(name=row.name,description=row.description,yaml=row.draft_yaml)
        catalog.seed_builtin_architecture(db,builtin_catalog.BUILTIN_ACTOR,resource_id,
            original.model_copy(update={"yaml":original.yaml+"\n"}))
        builtin_catalog.queue_ready_recipes(db)
        next_id = next(str(run.id) for run in db.scalars(select(Run))
                       if run.kind == RunKind.model_validation and run.status == RunStatus.queued)
    assert publish_fixture(next_id)[1].report.verdict == "valid"
    with store.session() as db:
        catalog.seed_builtin_architecture(db,builtin_catalog.BUILTIN_ACTOR,resource_id,original)
        builtin_catalog.queue_ready_recipes(db)
        assert len(catalog.revision_rows(db,"architecture",resource_id)) == 2
        assert len([row for row in db.scalars(select(Run)) if row.kind == RunKind.model_validation]) == 14


async def test_supervised_builtin_bootstrap_uses_real_parent_queue_and_child_receipts(client,tmp_path):
    """Run all seeded checks through actual supervised processes when OS IPC is available; fixtures do not substitute for this gate."""
    builtin_catalog.seed_blocks()
    builtin_catalog.seed_recipes()
    registry = Registry()
    registry.bind_loop(asyncio.get_running_loop())
    with WorkerManager(tmp_path/"runtime") as workers:
        runner = Runner(registry,{RunKind.model_validation:ValidationHandler()})
        runner.attach_workers(workers)
        await asyncio.wait_for(runner._drain_once(),300)
    with store.session() as db:
        runs = list(db.scalars(select(Run)))
        assert len(runs) == 13 and all(run.status == RunStatus.done for run in runs)
        assert len(list(db.scalars(select(ArchitectureRevision)))) == 4
