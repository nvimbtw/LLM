"""Real PostgreSQL catalog transactions, migration compatibility and retained evidence."""

from __future__ import annotations

import copy
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
import yaml

from llm.architecture import contracts
from llm.architecture.spec import (
    BlockDeclaration, DraftSubject, FinalTokenizer, PendingTokenizer, Recipe,
    ReceiptIdentity, ResolvedDefinition, RevisionSubject, RuntimeProfile,
    SourceManifestEntry, ValidationReceipt, canonical_digest,
)
from llm.architecture.wire import (
    AcceptRevision, ArchitectureCreate, ArchitectureDraftSave, BlockCreate,
    BlockDraftSave, EnqueueEvidence, MetadataPatch,
)
from llm import storage
from store import architecture_queries as catalog, queries
from store.schema import (
    ArchitectureDependency, ArchitectureRevision, BlockRevision, DefinitionEvidence,
    Model, ModelDefinition, ModelValidation, MutationRequest, Run, RunArchitecturePin,
    RunKind, RunStatus, TrainingRun, ValidationEvidence,
)
from tests.fixtures.postgres import database

pytestmark = pytest.mark.postgres


@pytest.fixture
def sessions(monkeypatch):
    """Migrate one uniquely owned database and provide independently committed transaction sessions."""
    with database() as url:
        monkeypatch.setenv("LLM_ENGINE_DATABASE_URL", url)
        command.upgrade(Config("alembic.ini"), "head")
        engine = create_engine(url)
        try:
            yield sessionmaker(engine, expire_on_commit=False)
        finally:
            engine.dispose()


@dataclass(frozen=True)
class Sample:
    """Retain the concrete IDs/bindings of a tiny accepted source and architecture fixture."""

    actor: uuid.UUID
    block_id: uuid.UUID
    block_revision: uuid.UUID
    architecture_id: uuid.UUID
    architecture_revision: uuid.UUID
    parameters: dict
    tokenizer: FinalTokenizer


def profile() -> RuntimeProfile:
    """Declare deterministic CPU evidence for the storage fixture without claiming model execution."""
    return RuntimeProfile(torch_version="2.13.0+cu130", sdk_version=1, device="cpu", precision="fp32",
                          capture_policy="torch-export-strict-v1", domains=[{
                              "batch_min": 1, "batch_max": 3, "sequence_min": 1, "sequence_max": 8,
                              "signatures": ["full_prefix"]}], modes=["train", "eval"])


def evidence(db, resource, dependencies: str, parameters: dict, tokenizer=None, *, subject=None):
    """Store typed fixture evidence and a real validation domain/job row for transaction tests."""
    run = queries.enqueue(db, kind=RunKind.model_validation, label="fixture validation", config={}, status=RunStatus.done)
    identity = ReceiptIdentity(subject=subject or DraftSubject(kind="draft", resource_id=resource.id,
                               draft_version=resource.draft_version, content_hash=resource.draft_hash),
                               dependency_digest=dependencies, parameter_digest=canonical_digest(parameters),
                               input_profile_digest=canonical_digest({}), tokenizer=tokenizer,
                               context_window=8 if tokenizer is not None else None, profile=profile())
    receipt = ValidationReceipt(schema_version=1, receipt_id=uuid.uuid4(), identity=identity,
                                identity_hash=canonical_digest(identity), verdict="valid", graph_manifest_hash="a" * 64)
    catalog.record_evidence(db, run.id, receipt)
    return run, receipt


def acceptance(resource, run, receipt) -> AcceptRevision:
    """Build exact draft/dependency expectations for one retained successful example receipt."""
    return AcceptRevision(expected_draft_version=resource.draft_version, validation_run_id=run.id,
                          expected_content_hash=resource.draft_hash,
                          expected_dependency_digest=receipt.identity.dependency_digest)


def block_declaration() -> BlockDeclaration:
    """Read the versioned synthetic composite declaration without importing custom Python."""
    return BlockDeclaration.model_validate(json.loads((contracts.CONTRACT_ROOT / "sdk/positive-legacy_transformer.json").read_text()))


@pytest.fixture
def sample(sessions):
    """Commit an accepted block/recipe with exact dependency pins and concrete tokenizer bindings."""
    actor = uuid.uuid4()
    tokenizer = FinalTokenizer(kind="final", tokenizer_id=uuid.uuid4(), fingerprint="b" * 64, vocabulary_size=32)
    with sessions.begin() as db:
        block = catalog.create_block(db, actor, BlockCreate(name="Fixture block", source="class StoredBlock: pass\n"))
        run, receipt = evidence(db, block, catalog.dependency_digest({}), {})
        revision = catalog.accept_block(db, block.id, acceptance(block, run, receipt), block_declaration())
        payload = contracts.decode_fixture_yaml((contracts.CONTRACT_ROOT / "recipes/positive-standard.yaml").read_text())
        payload["parameters"]["context_window"]["default"] = 8
        payload["modules"]["model"]["reference"] = catalog.revision_reference(revision).model_dump(mode="json")
        recipe = Recipe.model_validate(payload)
        parameters = {name: declaration.default for name, declaration in recipe.parameters.items()}
        architecture = catalog.create_architecture(db, actor, ArchitectureCreate(name="Fixture architecture", yaml=yaml.safe_dump(payload, sort_keys=False)))
        digest = catalog.dependency_digest(catalog.recipe_dependencies(recipe))
        run, receipt = evidence(db, architecture, digest, parameters, tokenizer)
        accepted = catalog.accept_architecture(db, architecture.id, acceptance(architecture, run, receipt), recipe)
        return Sample(actor, block.id, revision.id, architecture.id, accepted.id, parameters, tokenizer)


def enqueue_evidence(db, sample: Sample, *, exact: bool = False) -> EnqueueEvidence:
    """Read retained root acceptance and optionally bind fresh exact-profile evidence."""
    revision = db.get(ArchitectureRevision, sample.architecture_revision)
    receipt_id = None
    if exact:
        resource = catalog._locked_resource(db, "architecture", sample.architecture_id)
        _, receipt = evidence(db, resource, revision.dependency_digest, sample.parameters, sample.tokenizer,
                              subject=RevisionSubject(kind="revision", reference=catalog.revision_reference(revision)))
        receipt_id = receipt.receipt_id
    return EnqueueEvidence(architecture=catalog.revision_reference(revision), dependency_digest=revision.dependency_digest,
                            parameter_digest=canonical_digest(sample.parameters), acceptance_receipt_id=revision.acceptance_receipt_id,
                            tokenizer=sample.tokenizer if exact else PendingTokenizer(kind="pending", build_request_id=uuid.uuid4()),
                            exact_receipt_id=receipt_id, final_preflight_required=not exact)


def test_populated_legacy_migration_preserves_ids_and_nullable_definitions(monkeypatch):
    """Upgrade populated old history/models without replacing identities or inventing definitions."""
    with database() as url:
        monkeypatch.setenv("LLM_ENGINE_DATABASE_URL", url)
        command.upgrade(Config("alembic.ini"), "a4e79c1026bf")
        engine = create_engine(url)
        model_id, job_id = uuid.uuid4(), uuid.uuid4()
        try:
            with engine.begin() as connection:
                connection.execute(text("INSERT INTO models(id,name,published,created_at) VALUES (:id,'legacy',false,now())"), {"id": model_id})
                connection.execute(text("INSERT INTO execution_jobs(id,kind,status,label,rank,progress,created_at) VALUES (:id,'train','done','legacy',0,1,now())"), {"id": job_id})
                connection.execute(text("INSERT INTO training_runs(job_id,training_type,config,model_id) VALUES (:job,'pretrain','{}'::jsonb,:model)"), {"job": job_id, "model": model_id})
            command.upgrade(Config("alembic.ini"), "head")
            with sessionmaker(engine).begin() as db:
                assert db.get(Model, model_id).definition_id is None
                row = db.get(Run, job_id)
                assert row.kind == RunKind.train and row.ref_id == model_id
                assert row.training.definition_id is None and row.config == {}
                assert row.output == {"type": "model", "id": str(model_id)}
        finally:
            engine.dispose()


def test_invalid_drafts_and_read_only_builtins(sessions):
    """Saving bytes never executes/parses them; builtin mutation remains prohibited."""
    actor = uuid.uuid4()
    with sessions.begin() as db:
        block = catalog.create_block(db, actor, BlockCreate(name="Broken", source="def broken("))
        catalog.save_block_draft(db, block.id, BlockDraftSave(expected_version=1, source="still invalid ["))
        assert block.draft_source == "still invalid [" and block.draft_version == 2
        builtin = catalog.create_block(db, actor, BlockCreate(name="Builtin", source="invalid but stored"), builtin=True)
        with pytest.raises(catalog.CatalogError, match="read-only"):
            catalog.save_block_draft(db, builtin.id, BlockDraftSave(expected_version=1, source="changed"))


def test_concurrent_draft_save_has_one_winner(sessions, sample):
    """A PostgreSQL root lock makes equal expected-version saves mutually exclusive."""
    barrier = threading.Barrier(2)

    def save(value):
        """Compete with an independent transaction and return success or its stable refusal code."""
        try:
            with sessions.begin() as db:
                barrier.wait(timeout=5)
                catalog.save_block_draft(db, sample.block_id, BlockDraftSave(expected_version=1, source=value))
            return "saved"
        except catalog.CatalogError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(save, ("source one", "source two")))
    assert sorted(outcomes) == ["saved", "stale_version"]
    with sessions.begin() as db:
        row = catalog._locked_resource(db, "block", sample.block_id)
        assert row.draft_version == 2 and row.draft_source in {"source one", "source two"}
        assert db.get(BlockRevision, sample.block_revision).source == "class StoredBlock: pass\n"


def test_concurrent_acceptance_retains_one_revision(sessions):
    """Concurrent acceptance of identical validated bytes returns one immutable revision number."""
    with sessions.begin() as db:
        block = catalog.create_block(db, uuid.uuid4(), BlockCreate(name="Concurrent", source="class Concurrent: pass"))
        run, receipt = evidence(db, block, catalog.dependency_digest({}), {})
        block_id, request = block.id, acceptance(block, run, receipt)
    barrier = threading.Barrier(2)

    def accept(_):
        """Accept with a distinct database transaction after simultaneous admission."""
        with sessions.begin() as db:
            barrier.wait(timeout=5)
            return catalog.accept_block(db, block_id, request, block_declaration()).id

    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(accept, range(2)))
    assert ids[0] == ids[1]
    with sessions.begin() as db:
        assert db.scalar(select(func.count()).select_from(BlockRevision)) == 1


def test_archive_dependency_preserves_accepted_selection_and_snapshots(sessions, sample):
    """Dependency archive/draft edits never rewrite an accepted root or its queued source bytes."""
    with sessions.begin() as db:
        pins = enqueue_evidence(db, sample)
        queued = catalog.enqueue_pinned_training(db, actor_id=sample.actor, label="pinned", config={}, evidence=pins, parameters=sample.parameters)
        first = copy.deepcopy(db.get(RunArchitecturePin, queued.id).snapshot)
        catalog.update_metadata(db, "block", sample.block_id, MetadataPatch(expected_metadata_version=1, archived=True))
        catalog.save_block_draft(db, sample.block_id, BlockDraftSave(expected_version=1, source="new unaccepted bytes"))
        another = catalog.enqueue_pinned_training(db, actor_id=sample.actor, label="still pinned", config={}, evidence=pins, parameters=sample.parameters)
        assert db.get(RunArchitecturePin, another.id).snapshot == first
        assert first["dependencies"][0]["source"] == "class StoredBlock: pass\n"
        catalog.update_metadata(db, "architecture", sample.architecture_id, MetadataPatch(expected_metadata_version=1, archived=True))
        with pytest.raises(catalog.CatalogError, match="archived"):
            catalog.enqueue_pinned_training(db, actor_id=sample.actor, label="refused", config={}, evidence=pins, parameters=sample.parameters)


def test_archive_and_enqueue_serialize_on_root(sessions, sample):
    """Concurrent archive/enqueue has one legal ordering, never an unpinned accepted run."""
    with sessions.begin() as db:
        pins = enqueue_evidence(db, sample)
    barrier = threading.Barrier(2)

    def archive():
        """Commit archive while another connection attempts selection."""
        with sessions.begin() as db:
            barrier.wait(timeout=5)
            catalog.update_metadata(db, "architecture", sample.architecture_id, MetadataPatch(expected_metadata_version=1, archived=True))
        return "archived"

    def enqueue():
        """Commit a complete prior selection or expose archive conflict without creating a row."""
        try:
            with sessions.begin() as db:
                barrier.wait(timeout=5)
                return catalog.enqueue_pinned_training(db, actor_id=sample.actor, label="race", config={}, evidence=pins, parameters=sample.parameters).id
        except catalog.CatalogError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(archive), pool.submit(enqueue)]
        archived, selected = [future.result(timeout=10) for future in futures]
    assert archived == "archived"
    with sessions.begin() as db:
        pins_count = db.scalar(select(func.count()).select_from(RunArchitecturePin))
        assert pins_count == (0 if selected == "archived" else 1)
        if pins_count:
            assert db.get(RunArchitecturePin, selected).architecture_revision_id == sample.architecture_revision


def test_late_chain_error_rolls_back_all_queued_pins(sessions, sample):
    """One transaction cannot leave earlier model steps after a later invalid pin is found."""
    with sessions.begin() as db:
        pins = enqueue_evidence(db, sample)
    with pytest.raises(catalog.CatalogError, match="parameters"):
        with sessions.begin() as db:
            catalog.enqueue_pinned_training(db, actor_id=sample.actor, label="first", config={}, evidence=pins, parameters=sample.parameters)
            catalog.enqueue_pinned_training(db, actor_id=sample.actor, label="bad later", config={}, evidence=pins, parameters={"wrong": 1})
    with sessions.begin() as db:
        assert db.scalar(select(func.count()).select_from(RunArchitecturePin)) == 0
        assert db.scalar(select(func.count()).select_from(TrainingRun)) == 0


def test_http_chain_wake_observes_the_complete_postgres_commit(sessions, monkeypatch):
    """The route's wake callback sees all linked jobs through an independent PostgreSQL transaction."""
    import store
    from api.routes.runs import Step, queue_chain
    from api.runner import RUNNER

    monkeypatch.setattr(store, "_sessions", sessions)
    observed = []

    def observe_commit():
        """Read the full durable chain from a connection other than its enqueue transaction."""
        with sessions.begin() as db:
            observed.append([(run.id, run.depends_on) for run in queries.active_runs(db)])

    monkeypatch.setattr(RUNNER, "wake", observe_commit)
    result = queue_chain([Step(kind=RunKind.train, config={}),
                          Step(kind=RunKind.benchmark, config={})], owner=str(uuid.uuid4()))
    ids = [uuid.UUID(run["run_id"]) for run in result]
    assert observed == [[(ids[0], None), (ids[1], ids[0])]]


def test_http_chain_postgres_constraint_failure_leaves_no_jobs(sessions, monkeypatch):
    """A real later database constraint failure rolls back the earlier queue/domain insert and never wakes."""
    import store
    from api.routes.runs import Step, queue_chain
    from api.runner import RUNNER

    monkeypatch.setattr(store, "_sessions", sessions)
    enqueue = queries.enqueue
    inserted, wakes = [], []

    def fail_later_insert(*args, **kwargs):
        """Violate the actual nonnullable job label on the second insert in the route transaction."""
        if inserted:
            kwargs["label"] = None
        run = enqueue(*args, **kwargs)
        inserted.append(run.id)
        return run

    monkeypatch.setattr(queries, "enqueue", fail_later_insert)
    monkeypatch.setattr(RUNNER, "wake", lambda: wakes.append(True))
    steps = [Step(kind=RunKind.train, config={}), Step(kind=RunKind.benchmark, config={})]
    with pytest.raises(IntegrityError):
        queue_chain(steps, owner=str(uuid.uuid4()))
    assert len(inserted) == 1
    assert wakes == []
    with sessions.begin() as db:
        assert db.scalar(select(func.count()).select_from(Run)) == 0
        assert db.scalar(select(func.count()).select_from(TrainingRun)) == 0


def test_http_recipe_chain_pins_waiting_consumers_and_deduplicates_postgres(sessions, sample, monkeypatch):
    """The production route commits all inherited source pins and replays every original chain UUID on PostgreSQL."""
    import store
    from api.routes.runs import Step, queue_chain
    from api.runner import RUNNER

    monkeypatch.setattr(store, "_sessions", sessions)
    monkeypatch.setattr(RUNNER, "wake", lambda: None)
    operation = uuid.uuid4()
    steps = [Step(kind=RunKind.train, config={"architecture": {
                "revision_id": str(sample.architecture_revision), "parameters": {}}}),
             Step(kind=RunKind.train, config={})]
    first = queue_chain(steps, request_id=operation, owner=str(sample.actor), administration=True)
    again = queue_chain(steps, request_id=operation, owner=str(sample.actor), administration=True)
    assert [value["run_id"] for value in again] == [value["run_id"] for value in first]
    with sessions.begin() as db:
        pins = [db.get(RunArchitecturePin, uuid.UUID(value["run_id"])) for value in first]
        assert pins[0].snapshot == pins[1].snapshot
        assert pins[0].evidence == pins[1].evidence
        assert db.scalar(select(func.count()).select_from(TrainingRun)) == 2


def test_actor_request_retries_are_atomic_and_conflicts_do_not_reexecute(sessions):
    """Actual PostgreSQL unique-key blocking commits one identical actor/request effect."""
    actor, request_id = uuid.uuid4(), uuid.uuid4()
    barrier = threading.Barrier(2)
    executed = []

    def attempt(_):
        """Deduplicate the same operation from two independent committing sessions."""
        with sessions.begin() as db:
            barrier.wait(timeout=5)

            def execute():
                """Create one real queue row and return its stable logical operation identity."""
                run = queries.enqueue(db, kind=RunKind.train, label="once", config={})
                executed.append(run.id)
                return catalog.MutationOutcome("training", run.id)

            return catalog.idempotent_mutation(db, actor_id=actor, request_id=request_id,
                        operation="enqueue", payload={"name": "same"}, execute=execute)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert results[0] == results[1] and len(executed) == 1
    with pytest.raises(catalog.CatalogError, match="different arguments"):
        with sessions.begin() as db:
            catalog.idempotent_mutation(db, actor_id=actor, request_id=request_id, operation="enqueue",
                                       payload={"name": "different"}, execute=lambda: pytest.fail("reexecuted"))
    with sessions.begin() as db:
        assert db.scalar(select(func.count()).select_from(Run)) == 1
        assert db.scalar(select(func.count()).select_from(MutationRequest)) == 1


def test_job_pruning_cannot_remove_accepted_waiting_or_package_evidence(sessions, sample):
    """Real FK retention survives validation history pruning and a bounded unused-evidence sweep."""
    with sessions.begin() as db:
        pins = enqueue_evidence(db, sample, exact=True)
        prerequisite = queries.enqueue(db, kind=RunKind.train, label="parent", config={})
        waiting = catalog.enqueue_pinned_training(db, actor_id=sample.actor, label="waiting", config={}, evidence=pins,
                                                 parameters=sample.parameters, depends_on=prerequisite.id)
        assert waiting.status == RunStatus.waiting
        revision = db.get(ArchitectureRevision, sample.architecture_revision)
        closure = catalog.dependency_closure(db, revision.id)
        definition = ResolvedDefinition(schema_version=1, architecture=catalog.revision_reference(revision),
             recipe=Recipe.model_validate(revision.recipe), dependencies=[SourceManifestEntry(reference=item.reference,
             path=f"sources/{item.reference.revision_id}.py", bytes=len(item.source.encode()), sdk_version=1) for item in closure],
             parameters=sample.parameters, tokenizer=sample.tokenizer, context_window=8, aliases=[], sdk_version=1)
        saved = catalog.record_definition(db, definition, (pins.exact_receipt_id,))
        assert catalog.record_definition(db, definition, (pins.exact_receipt_id,)).id == saved.id
        waiting.training.definition_id = saved.id
        model = queries.upsert_model(db, model_id=uuid.uuid4(), name="retained", definition_id=saved.id)
        resource = catalog._locked_resource(db, "architecture", sample.architecture_id)
        unused_run, unused = evidence(db, resource, revision.dependency_digest, sample.parameters, sample.tokenizer)
        for run in list(db.scalars(select(Run).where(Run.kind == RunKind.model_validation))):
            assert queries.delete_run(db, run.id)
        assert catalog.prune_unreferenced_evidence(db) == 1
        assert db.get(ValidationEvidence, unused.receipt_id) is None
        assert db.get(ValidationEvidence, pins.exact_receipt_id) is not None
        assert db.get(BlockRevision, sample.block_revision) is not None
        assert db.get(ModelDefinition, saved.id) is not None
        assert db.get(Model, model.id).definition_id == saved.id
        assert db.scalar(select(func.count()).select_from(DefinitionEvidence)) == 1


def test_postgres_rejects_bad_dependency_types_and_destructive_references(sessions, sample):
    """Production FKs/type checks reject invalid dual pins and deletion of retained sources/evidence."""
    with sessions.begin() as db:
        receipt_id = db.get(ArchitectureRevision, sample.architecture_revision).acceptance_receipt_id
    for values in ({"block_revision_id": sample.block_revision, "child_architecture_revision_id": sample.architecture_revision},
                   {"block_revision_id": None, "child_architecture_revision_id": None}):
        with pytest.raises(IntegrityError):
            with sessions.begin() as db:
                db.add(ArchitectureDependency(architecture_revision_id=sample.architecture_revision, instance_path="invalid", **values))
    for statement, identity in (("DELETE FROM block_revisions WHERE id=:id", sample.block_revision),
                                ("DELETE FROM validation_evidence WHERE receipt_id=:id", receipt_id)):
        with pytest.raises(IntegrityError):
            with sessions.begin() as db:
                db.execute(text(statement), {"id": identity})


def test_postgres_immutable_source_trigger_and_validation_facade(sessions, sample):
    """Raw SQL cannot update accepted source; the validation output stays its own job UUID."""
    with pytest.raises(IntegrityError, match="immutable"):
        with sessions.begin() as db:
            db.execute(text("UPDATE block_revisions SET source='changed' WHERE id=:id"), {"id": sample.block_revision})
    with sessions.begin() as db:
        run = queries.enqueue(db, kind=RunKind.model_validation, label="validation", config={"mode": "inspect"})
        assert run.output == {"type": "validation", "id": str(run.id)}
        run.result = {"verdict": "inspection_only"}
        assert db.get(ModelValidation, run.id).result == {"verdict": "inspection_only"}
        with pytest.raises(ValueError, match="UUID"):
            run.ref_id = uuid.uuid4()


def test_new_storage_paths_validate_identity_without_creating_directories(tmp_path, monkeypatch):
    """Only UUID/digest paths can address new source/runtime artifacts; path helpers own no publication."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    run_id = uuid.uuid4()
    assert storage.validation_dir(run_id) == tmp_path / "artifacts/validations" / str(run_id)
    assert storage.definition_dir("a" * 64) == tmp_path / "artifacts/definitions" / ("a" * 64)
    assert storage.runtime_session_dir(run_id) == tmp_path / "artifacts/.runtime" / str(run_id)
    assert storage.runtime_lock_path() == tmp_path / "artifacts/.runtime/engine.lock"
    assert list(tmp_path.iterdir()) == []
    for invalid in ("../private", "A" * 64, "a" * 63):
        with pytest.raises(ValueError):
            storage.definition_dir(invalid)


def test_stale_acceptance_and_provisional_evidence_cannot_replace_revision(sessions, sample):
    """Acceptance rechecks version and fully bound evidence in the same production transaction."""
    with sessions.begin() as db:
        resource = catalog._locked_resource(db, "architecture", sample.architecture_id)
        existing = db.get(ArchitectureRevision, sample.architecture_revision)
        run, receipt = evidence(db, resource, existing.dependency_digest, sample.parameters, sample.tokenizer)
        request = acceptance(resource, run, receipt)
        recipe = Recipe.model_validate(existing.recipe)
        catalog.save_architecture_draft(db, resource.id, ArchitectureDraftSave(expected_version=1, yaml=resource.draft_yaml + "\n"))
    with pytest.raises(catalog.CatalogError, match="draft changed"):
        with sessions.begin() as db:
            catalog.accept_architecture(db, sample.architecture_id, request, recipe)
    with sessions.begin() as db:
        resource = catalog._locked_resource(db, "architecture", sample.architecture_id)
        pending = PendingTokenizer(kind="pending", build_request_id=uuid.uuid4())
        run = queries.enqueue(db, kind=RunKind.model_validation, label="pending", config={}, status=RunStatus.done)
        identity = receipt.identity.model_copy(update={"subject": DraftSubject(kind="draft", resource_id=resource.id,
                         draft_version=resource.draft_version, content_hash=resource.draft_hash), "tokenizer": pending})
        provisional = ValidationReceipt(schema_version=1, receipt_id=uuid.uuid4(), identity=identity,
                         identity_hash=canonical_digest(identity), verdict="provisional", graph_manifest_hash="a" * 64)
        catalog.record_evidence(db, run.id, provisional)
        with pytest.raises(catalog.CatalogError, match="not valid"):
            catalog.accept_architecture(db, resource.id, acceptance(resource, run, provisional), recipe)
        assert db.scalar(select(func.count()).select_from(ArchitectureRevision)) == 1


def test_nested_architecture_closure_keeps_archived_child_sources(sessions, sample):
    """Accepted subarchitectures retain transitive source pins across child resource archive."""
    with sessions.begin() as db:
        child = db.get(ArchitectureRevision, sample.architecture_revision)
        payload = copy.deepcopy(child.recipe)
        payload["modules"]["model"]["reference"] = catalog.revision_reference(child).model_dump(mode="json")
        payload["modules"]["model"]["parameters"] = {name: f"${{{name}}}" for name in sample.parameters}
        recipe = Recipe.model_validate(payload)
        parent = catalog.create_architecture(db, sample.actor, ArchitectureCreate(name="Parent", yaml=yaml.safe_dump(payload)))
        digest = catalog.dependency_digest(catalog.recipe_dependencies(recipe))
        run, receipt = evidence(db, parent, digest, sample.parameters, sample.tokenizer)
        revision = catalog.accept_architecture(db, parent.id, acceptance(parent, run, receipt), recipe)
        catalog.update_metadata(db, "architecture", sample.architecture_id, MetadataPatch(expected_metadata_version=1, archived=True))
        pins = EnqueueEvidence(architecture=catalog.revision_reference(revision), dependency_digest=digest,
                 parameter_digest=canonical_digest(sample.parameters), acceptance_receipt_id=receipt.receipt_id,
                 tokenizer=PendingTokenizer(kind="pending", build_request_id=uuid.uuid4()), exact_receipt_id=None,
                 final_preflight_required=True)
        queued = catalog.enqueue_pinned_training(db, actor_id=sample.actor, label="nested", config={}, evidence=pins, parameters=sample.parameters)
        snapshot = db.get(RunArchitecturePin, queued.id).snapshot
        assert len(snapshot["dependencies"]) == 2
        assert {item["reference"]["kind"] for item in snapshot["dependencies"]} == {"block", "architecture"}


def test_idempotency_is_actor_scoped_and_outlives_pruned_execution(sessions):
    """The same request UUID can belong to two actors while one actor's old outcome survives pruning."""
    request_id, actor = uuid.uuid4(), uuid.uuid4()
    with sessions.begin() as db:
        def execute():
            """Create one completed execution to exercise post-history retry retention."""
            run = queries.enqueue(db, kind=RunKind.train, label="dedupe", config={}, status=RunStatus.done)
            return catalog.MutationOutcome("training", run.id)

        first = catalog.idempotent_mutation(db, actor_id=actor, request_id=request_id,
                    operation="enqueue", payload={}, execute=execute)
        second = catalog.idempotent_mutation(db, actor_id=uuid.uuid4(), request_id=request_id,
                    operation="enqueue", payload={}, execute=execute)
        assert first.id != second.id
        assert queries.delete_run(db, first.id)
        retried = catalog.idempotent_mutation(db, actor_id=actor, request_id=request_id,
                    operation="enqueue", payload={}, execute=lambda: pytest.fail("reexecuted pruned operation"))
        assert retried == first and db.get(Run, first.id) is None
        assert db.scalar(select(func.count()).select_from(MutationRequest)) == 2


def test_unfinished_validation_cannot_publish_authorization_evidence(sessions, sample):
    """A valid-looking payload cannot authorize acceptance before durable validation completion."""
    with sessions.begin() as db:
        resource = catalog._locked_resource(db, "architecture", sample.architecture_id)
        retained = db.get(ValidationEvidence, db.get(ArchitectureRevision, sample.architecture_revision).acceptance_receipt_id)
        receipt = ValidationReceipt.model_validate(retained.receipt).model_copy(update={"receipt_id": uuid.uuid4()})
        run = queries.enqueue(db, kind=RunKind.model_validation, label="not completed", config={})
        with pytest.raises(catalog.CatalogError, match="completed validation"):
            catalog.record_evidence(db, run.id, receipt)
        assert db.get(ValidationEvidence, receipt.receipt_id) is None
