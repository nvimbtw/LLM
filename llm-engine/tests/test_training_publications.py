"""Data-only parent package/adoption transactions; synthetic tensors/graphs do not certify numerical or process execution."""

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

import store
from api import recovery
from api import training_publications as publications
from llm import storage
from llm.architecture import package, package_transfer
from llm.architecture.graphs import GraphArtifacts, export_guards
from llm.architecture.package_integrity import encode_json
from llm.architecture.spec import (
    PendingTokenizer, ReceiptIdentity, Recipe, ResolvedDefinition, RevisionSubject,
    SourceManifestEntry, ValidationReceipt, canonical_digest,
)
from llm.architecture.tokenizer_binding import binding_from_bytes
from llm.config import TrainConfig
from llm.runtime.artifacts import ArtifactStore
from llm.runtime.protocol import ArtifactReady, RuntimeResources
from llm.train.model_preflight import training_profile
from llm.train.modular_records import TrainingGeneration, TrainingPublication
from store import architecture_queries as catalog, queries as q
from store.schema import ArchitectureRevision, DefinitionEvidence, Model, ModelDefinition, Run, RunKind, RunStatus, ValidationEvidence
from tests.test_architecture_store import enqueue_evidence, sample


@pytest.fixture
def sessions(store_session):
    """Use isolated actual SQLite transactions; the same lifecycle has a separate runnable PostgreSQL gate."""
    return store._sessions


@dataclass(frozen=True)
class PublishedJob:
    """Own explicitly synthetic evidence and immutable accepted fixture source for parent-only transaction tests."""

    job_id: UUID
    config: TrainConfig
    definition: ResolvedDefinition
    receipt: ValidationReceipt
    files: dict[str, bytes]

    def write(self, root: Path, *, index: int = 1, step: int = 1, final: bool = False, config: TrainConfig | None = None):
        """Seal complete parent-readable fixture bytes without importing source or serializing real tensors."""
        from llm.architecture.spec import DefinitionPackage

        model_id = UUID(self.config.model_name) if final else uuid4()
        generation = TrainingGeneration(model_id=model_id, index=0 if final else index, step=3 if final else step,
            total_steps=3, epoch=1, stage=0, stage_name=None, path="model.pt" if final else "train_state.pt",
            loss=1.0, architecture={"ctx_window": 8}, final=final)
        result = {"recovery": not final, "checkpoint": generation.model_dump(mode="json"),
            "definition_id": canonical_digest(self.definition), "vocab": self.definition.tokenizer.vocabulary_size,
            "ctx_window": 8, "parameters": 123}
        metadata = TrainingPublication(job_id=self.job_id, config=config or self.config, generation=generation, result=result)
        files = {**self.files, "meta.json": metadata.model_dump_json().encode(), "model.pt": b"synthetic tensor bytes"}
        if not final:
            files["train_state.pt"] = b"synthetic optimizer bytes"
        manifest = DefinitionPackage(format="slate.modular-model", format_version=1,
            definition_id=canonical_digest(self.definition), definition=self.definition, receipt_ids=(self.receipt.receipt_id,),
            graph_hashes=tuple(name[7:-5] for name in files if name.startswith("graphs/")), parameter_precision="fp32",
            training_state_path=None if final else "train_state.pt", optimizer_parameter_order=() if final else ("fixture.weight",))
        destination = root / str(model_id)
        package.publish(destination, manifest, files)
        return package.verify(destination)


@pytest.fixture
def published_job(sessions, sample, tmp_path, monkeypatch):
    """Queue one accepted pin and construct labeled synthetic receipt graphs, never a purported numerical check."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "publication-data"))
    raw = Tokenizer(WordLevel(vocab={"<|endoftext|>": 0, "one": 1}, unk_token="<|endoftext|>")).to_str().encode()
    binding = binding_from_bytes(uuid4(), raw)
    job_id = uuid4()
    cfg = TrainConfig(model_name=str(uuid4()), batch_size=2, precision="fp32", compile_model=False,
                      architecture={"revision_id": sample.architecture_revision, "parameters": {}})
    with sessions.begin() as db:
        revision = db.get(ArchitectureRevision, sample.architecture_revision)
        recipe = Recipe.model_validate(revision.recipe)
        closure = catalog.dependency_closure(db, revision.id)
        entries = tuple(SourceManifestEntry(reference=item.reference, path=f"sources/{item.reference.revision_id}.py",
            bytes=len(item.source.encode()), sdk_version=1) for item in closure)
        definition = ResolvedDefinition(schema_version=1, architecture=catalog.revision_reference(revision), recipe=recipe,
            dependencies=entries, parameters=sample.parameters, tokenizer=binding, context_window=8, aliases=recipe.aliases, sdk_version=1)
        profile = training_profile(cfg, 8, "cpu")
        identity = ReceiptIdentity(subject=RevisionSubject(kind="revision", reference=definition.architecture),
            dependency_digest=revision.dependency_digest, parameter_digest=canonical_digest(sample.parameters),
            input_profile_digest=canonical_digest({"tensor_inputs": {}, "domains": [domain.model_dump(mode="json") for domain in profile.domains]}),
            tokenizer=binding, context_window=8, profile=profile)
        graphs = GraphArtifacts("")
        for mode in profile.modes:
            graphs.add(view="operations", mode=mode, signature="full_prefix", scope="root", strict=True, guards=(),
                nodes=({"id": "fixture", "kind": "input", "label": "Synthetic parent transaction fixture", "occurrence_path": "fixture"},), edges=())
        identity = identity.model_copy(update={"profile": profile.model_copy(update={"export_guards": export_guards(graphs.entries)})})
        graphs.identity_hash = canonical_digest(identity)
        manifest_hash, blobs = graphs.finish()
        receipt = ValidationReceipt(schema_version=1, receipt_id=uuid4(), identity=identity,
            identity_hash=canonical_digest(identity), verdict="valid", graph_manifest_hash=manifest_hash)
        evidence = enqueue_evidence(db, sample).model_copy(update={"tokenizer": PendingTokenizer(kind="pending", build_request_id=job_id)})
        prepared = catalog.prepare_training_pin(db, evidence, sample.parameters)
        run = q.enqueue(db, id=job_id, kind=RunKind.train, label="parent fixture", status=RunStatus.running,
                        config={**cfg.model_dump(mode="json", exclude_unset=True), "display_name": "Fixture run"})
        catalog.attach_training_pin(db, run, prepared)
        files = {"architecture.yaml": revision.yaml.encode(), "tokenizer.json": raw,
                 f"receipts/{receipt.receipt_id}.json": encode_json(receipt)}
        files.update({entry.path: item.source.encode() for entry, item in zip(entries, closure, strict=True)})
        files.update({f"graphs/{digest}.json": source.encode() for digest, source in blobs.items()})
    return PublishedJob(job_id, cfg, definition, receipt, files)


def granted(checked, tmp_path):
    """Create a real bounded archive/grant/notice while leaving process acknowledgement to the actual supervisor tests."""
    archive = package_transfer.write_archive(checked.directory, tmp_path / f"{checked.directory.name}.tar")
    artifacts = ArtifactStore(tmp_path / "grants", RuntimeResources(process_address_bytes=16 << 30,
        tree_resident_bytes=3 << 30, tree_memory_enforcement="preflight_and_monitor", device="cpu",
        device_allocator_bytes=0, artifact_bytes=16 << 20, total_artifact_bytes=64 << 20))
    value = publications.metadata(checked)
    notice = ArtifactReady(type="artifact_ready", publication_id=uuid4(),
        artifact=artifacts.stage_file(archive, "model_package_v1"), completed_step=value.generation.step, final=value.generation.final)
    return notice, artifacts


def test_parent_adopts_registers_retains_and_replays_complete_generations(published_job, tmp_path, monkeypatch):
    """Parent adoption retains one latest recovery plus final output, with exact evidence and no source/tensor execution."""
    job = published_job

    def forbidden(*_args, **_kwargs):
        """Detect any source import or tensor unpickling during parent publication."""
        pytest.fail("parent adoption executed source or deserialized tensors")

    monkeypatch.setattr("llm.architecture.source.SourceLoader.load", forbidden)
    monkeypatch.setattr("torch.load", forbidden)
    frames = []
    for options in ({}, {"index": 2, "step": 2}, {"final": True}):
        checked = job.write(tmp_path / "worker", **options)
        notice, artifacts = granted(checked, tmp_path)
        frame = publications.adopt(notice, artifacts, job_id=job.job_id)
        assert publications.adopt(notice, artifacts, job_id=job.job_id) == frame
        frames.append(frame)
        artifacts.remove(notice.artifact)
    with store.session() as db:
        assert db.get(Model, UUID(frames[0].model_id)) is None
        assert not storage.model_dir(frames[0].model_id).exists()
        recovery = db.get(Model, UUID(frames[1].model_id))
        final = db.get(Model, UUID(frames[2].model_id))
        assert recovery.name == "Fixture run #2" and recovery.parent_id == UUID(job.config.model_name)
        assert final.name == "Fixture run" and final.checkpoint_index is None and final.context == 8
        assert final.definition_id == canonical_digest(job.definition) == db.get(Run, job.job_id).training.definition_id
        assert db.scalar(select(func.count()).select_from(DefinitionEvidence)) == 1
        assert catalog.evidence_for_job(db, job.job_id).receipt_id == job.receipt.receipt_id
        final.name, final.published = "Human display edit", True
    assert publications.register(package.verify(storage.model_dir(job.config.model_name))) == frames[-1]
    with store.session() as db:
        final = db.get(Model, UUID(job.config.model_name))
        assert final.name == "Human display edit" and final.published
        db.get(Run, job.job_id).status = RunStatus.done
        q.delete_run(db, job.job_id)
        assert catalog.prune_unreferenced_evidence(db) == 0
        assert db.get(ModelDefinition, final.definition_id) is not None
        assert db.get(ValidationEvidence, job.receipt.receipt_id) is not None
        assert catalog.record_training_evidence(db, job.job_id, job.receipt, job.definition).receipt_id == job.receipt.receipt_id


def test_parent_index_failure_preserves_adoptable_package_and_rolls_back_definition_evidence(published_job, tmp_path, monkeypatch):
    """A post-rename index failure cannot ACK or lose the complete orphan; interrupted-job adoption is idempotent."""
    job = published_job
    checked = job.write(tmp_path / "worker")
    notice, artifacts = granted(checked, tmp_path)

    def failed_index(*_args, **_kwargs):
        """Interrupt the real transaction after evidence/definition insertion but before model registration."""
        raise RuntimeError("model index unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(q, "upsert_model", failed_index)
        with pytest.raises(RuntimeError, match="model index unavailable"):
            publications.adopt(notice, artifacts, job_id=job.job_id)
    destination = storage.model_dir(checked.directory.name)
    assert destination.is_dir()
    with store.session() as db:
        assert db.get(Model, UUID(destination.name)) is None
        assert db.get(ValidationEvidence, job.receipt.receipt_id) is None
        assert db.get(ModelDefinition, checked.manifest.definition_id) is None
        db.get(Run, job.job_id).status = RunStatus.failed
    recovery.reconcile()
    frame = publications.register(package.verify(destination))
    assert frame.model_id == destination.name
    with store.session() as db:
        assert db.get(Model, UUID(frame.model_id)).definition_id == checked.manifest.definition_id
        assert db.get(Run, job.job_id).status == RunStatus.failed


@pytest.mark.parametrize("mismatch", ["job", "position", "configuration"])
def test_parent_refuses_foreign_or_changed_publications_before_model_rename(published_job, tmp_path, mismatch):
    """A valid package cannot authorize a different request, completed boundary or immutable queued configuration."""
    job = published_job
    cfg = job.config.model_copy(update={"lr": 0.2}) if mismatch == "configuration" else None
    checked = job.write(tmp_path / "worker", config=cfg)
    notice, artifacts = granted(checked, tmp_path)
    if mismatch == "position":
        notice = notice.model_copy(update={"completed_step": 2})
    with pytest.raises(ValueError, match="differs"):
        publications.adopt(notice, artifacts, job_id=uuid4() if mismatch == "job" else job.job_id)
    assert not storage.model_dir(checked.directory.name).exists()
    with store.session() as db:
        assert db.get(ValidationEvidence, job.receipt.receipt_id) is None
        assert db.get(Model, UUID(checked.directory.name)) is None


def test_training_evidence_cannot_bypass_admission_or_validation_completion(published_job):
    """Unadmitted training and validation-only APIs cannot retain a new training receipt; matching admitted work can."""
    job = published_job
    with store.session() as db:
        run = db.get(Run, job.job_id)
        run.status = RunStatus.queued
        with pytest.raises(catalog.CatalogError, match="admitted pinned job"):
            catalog.record_training_evidence(db, job.job_id, job.receipt, job.definition)
        run.status = RunStatus.done
        with pytest.raises(catalog.CatalogError, match="completed validation job"):
            catalog.record_evidence(db, job.job_id, job.receipt)
        run.status = RunStatus.running
        catalog.record_training_evidence(db, job.job_id, job.receipt, job.definition)
        assert catalog.record_training_evidence(db, job.job_id, job.receipt, job.definition).receipt_id == job.receipt.receipt_id
        changed_definition = job.definition.model_copy(update={"tokenizer": job.definition.tokenizer.model_copy(update={"tokenizer_id": uuid4()})})
        with pytest.raises(catalog.CatalogError, match="saved definition"):
            catalog.record_training_evidence(db, job.job_id, job.receipt, changed_definition)
        changed = job.receipt.model_copy(update={"receipt_id": uuid4()})
        with pytest.raises(catalog.CatalogError, match="already owns different"):
            catalog.record_training_evidence(db, job.job_id, changed, job.definition)


def test_startup_discards_only_unregistered_generations_older_than_durable_output(published_job, tmp_path):
    """A later durable generation remains authoritative when startup finds an older post-rename orphan."""
    job = published_job
    older = job.write(tmp_path / "worker")
    archive = package_transfer.write_archive(older.directory, tmp_path / "orphan.tar")
    old_destination = storage.model_dir(older.directory.name)
    package_transfer.materialize(archive, old_destination)
    newer = job.write(tmp_path / "worker", index=2, step=2)
    notice, artifacts = granted(newer, tmp_path)
    frame = publications.adopt(notice, artifacts, job_id=job.job_id)
    recovery.reconcile()
    assert not old_destination.exists()
    assert storage.model_dir(frame.model_id).exists()
    with store.session() as db:
        assert db.get(Model, UUID(older.directory.name)) is None
        assert db.get(Model, UUID(frame.model_id)).checkpoint_step == 2
