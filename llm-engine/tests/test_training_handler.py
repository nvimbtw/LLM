"""Parent handler/queue contracts with explicit synthetic session fixtures; real process acceptance remains separate."""

import asyncio
from contextlib import contextmanager
from pathlib import Path
import shutil
from threading import Event
from uuid import UUID, uuid4

import numpy as np
import pytest
import jwt
from sqlalchemy import delete

import store
from api import handlers, training_admission, training_handler, training_jobs
from api.runner import Runner
from api.runs import Registry
from api.worker import RegisteredCheckpoint
from llm import storage
from llm.architecture import package_transfer
from llm.architecture.package_integrity import encode_json
from llm.architecture.spec import canonical_digest
from llm.config import TrainConfig
from llm.corpus import vocab
from llm.corpus.spans import DocSpan, Spans
from llm.model import device
from llm.orchestrate.modular_data import PreparedData
from llm.runtime.artifacts import ArtifactStore
from llm.runtime.protocol import ArtifactReady, Completed
from llm.runtime.supervisor import CleanupUncertain
from llm.train.data_identity import fingerprint
from llm.train.modular_records import TrainingPreparation
from llm.train.plan import DataPlan
from llm.train.progress import Progress
from store import architecture_queries as catalog, queries as q
from store.schema import Run, RunArchitecturePin, RunKind, RunStatus
from tests.test_architecture_visibility import SECRET, headers
from tests.test_training_publications import published_job, sample, sessions
from tests.test_validation_queue import client


class FixtureSession:
    """Materialize honest synthetic package/result fixtures through the real parent adopter without executing a numerical child."""

    def __init__(self, root, policy, job, events):
        """Own real bounded artifact files for the fixture session's single request."""
        self.directory, self.job, self.events = root, job, events
        self.artifacts = ArtifactStore(root / "artifacts", policy.resources)

    def run(self, command, *, request_id, adopt, cancel):
        """Check actual private input grants and drive complete real package adoption before returning synthetic summary bytes."""
        prep = TrainingPreparation.model_validate_json(self.artifacts.read_bytes(command.config))
        assert command.job_id == request_id == self.job.job_id
        assert prep.validation.run_id == request_id and command.pins is None
        assert command.corpus.size_bytes == prep.data.tokens * 2
        assert not cancel.is_set()
        self.artifacts.verify(command.corpus)
        self.artifacts.verify(prep.tokenizer)
        self.artifacts.verify(prep.data.spans)
        frame = Progress(step=3, total_steps=3, epoch=1, total_epochs=1, chunk=3, total_chunks=3,
                         loss=1.0, lr=0.001, elapsed_secs=1.0)
        telemetry = self.artifacts.write_bytes(training_handler.PROGRESS.dump_json(frame), "training_progress_v1")
        adopt(ArtifactReady(type="artifact_ready", publication_id=uuid4(), artifact=telemetry, completed_step=3, final=False), self.artifacts)
        self.artifacts.remove(telemetry)
        checked = self.job.write(self.directory / "synthetic-output", final=True)
        archive = package_transfer.write_archive(checked.directory, self.directory / "output.tar")
        grant = self.artifacts.stage_file(archive, "model_package_v1")
        adopt(ArtifactReady(type="artifact_ready", publication_id=uuid4(), artifact=grant, completed_step=3, final=True), self.artifacts)
        self.artifacts.remove(grant)
        self.events.append("registered")
        result = {"model_id": self.job.config.model_name, "definition_id": canonical_digest(self.job.definition),
                  "steps": 3, "total_steps": 3, "data_fingerprint": prep.data.fingerprint}
        return Completed(type="completed", outcome="ok", result=self.artifacts.write_bytes(encode_json(result), "training_result_v1"))


class FixtureManager:
    """Exercise the real parent session interface with labeled synthetic output, never a forged process certificate."""

    def __init__(self, root, job, *, fail_cleanup=False):
        """Retain one synthetic adapter scenario and its observable cleanup boundary."""
        self.root, self.job, self.fail_cleanup = root, job, fail_cleanup
        root.mkdir()
        self.events = []

    def evict_for_queue(self):
        """Record that queue priority was requested before admission."""
        self.events.append("evicted")

    @contextmanager
    def session(self, lease, policy, factory):
        """Require the actual active parent lease and let cleanup failure propagate after durable adoption."""
        lease.require_active()
        with pytest.raises(device.Busy):
            with lease.gate.admit():
                pytest.fail("handler acquired another device owner")
        assert factory == "llm.train.modular_worker:TrainingBackend"
        self.events.append("session")
        try:
            yield FixtureSession(self.root / "session", policy, self.job, self.events)
        finally:
            with store.session() as db:
                assert db.get(Run, self.job.job_id).status == RunStatus.running
            self.events.append("cleanup")
            if self.fail_cleanup:
                raise CleanupUncertain("fixture cleanup refusal")


@pytest.fixture
def prepared_parent(published_job, monkeypatch):
    """Provide real numeric/tokenizer/span files at the parent preparation seam without claiming authenticated transport."""
    observed = []
    job = published_job

    def prepare(cfg, **kwargs):
        """Write bounded owned input fixtures and retain the actual actor/context/admission passed by the production handler."""
        observed.append(kwargs)
        assert kwargs["context_window"] == 8 and kwargs["budget"].max_corpus_bytes > 98
        directory = kwargs["directory"]
        corpus = np.arange(49, dtype=np.uint16) % 2
        path, span_path, token_path = directory / "corpus.bin", directory / "spans.npy", directory / "tokenizer.json"
        corpus.tofile(path)
        documents = Spans.of([DocSpan("fixture", 0, len(corpus))])
        documents.save(span_path)
        raw = job.files["tokenizer.json"]
        token_path.write_bytes(raw)
        binding = job.definition.tokenizer
        build = storage.tokenizer_build_dir(str(binding.tokenizer_id))
        build.mkdir(parents=True, exist_ok=True)
        storage.tokenizer_file(build).write_bytes(raw)
        storage.tokenizer_meta_path(str(binding.tokenizer_id)).write_bytes(encode_json({
            "fingerprint": binding.fingerprint[:vocab.FINGERPRINT_CHARS], "vocab_size": 2, "merges": 0}))
        plan = DataPlan(documents=documents, eot_id=0, shuffle=cfg.shuffle, seed=cfg.seed)
        return PreparedData(path, span_path, token_path, binding, len(corpus), ("fixture",), (), 0, "fixture",
                            fingerprint(corpus, plan=plan), None, cfg.shuffle, cfg.seed)

    monkeypatch.setattr(training_handler.modular_data, "prepare", prepare)
    monkeypatch.setattr(training_handler.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training_admission, "capacity", lambda path: training_admission.Capacity(8 << 30, 4 << 30))
    return observed


@pytest.mark.parametrize("outcome", ["done", "cleanup_failure", "cancel_before_terminal"])
async def test_default_queue_uses_one_parent_lease_and_waits_for_cleanup_before_terminal_state(
        published_job, prepared_parent, tmp_path, monkeypatch, outcome):
    """Real queue/registration/telemetry adapters preserve cleanup ordering; synthetic session success never substitutes for process evidence."""
    registry = Registry()
    registry.bind_loop(asyncio.get_running_loop())
    gate = device.DeviceGate()
    monkeypatch.setattr(device, "GATE", gate)
    manager = FixtureManager(tmp_path / "runtime", published_job, fail_cleanup=outcome == "cleanup_failure")
    if outcome == "cancel_before_terminal":
        completed_result = training_handler.completed_result

        def cancelled_result(*args):
            """Model a durable cancellation arriving after cleanup but before Runner acquires its terminal commit lock."""
            result = completed_result(*args)
            with store.session() as db:
                db.get(Run, published_job.job_id).status = RunStatus.cancelling
            return result

        monkeypatch.setattr(training_handler, "completed_result", cancelled_result)

    def forbidden(*args, **kwargs):
        """Detect any legacy execution, duplicate registration, parent tensor load or custom source import."""
        pytest.fail("parent crossed the supervised training boundary")

    monkeypatch.setattr(handlers.train_pipeline, "run", forbidden)
    monkeypatch.setattr(handlers, "record_checkpoint", forbidden)
    monkeypatch.setattr("torch.load", forbidden)
    monkeypatch.setattr("llm.architecture.source.SourceLoader.load", forbidden)
    actor = uuid4()
    with store.session() as db:
        run = db.get(Run, published_job.job_id)
        run.status, run.owner_id = RunStatus.queued, actor
    runner = Runner(registry, {RunKind.train: handlers.train})
    runner.attach_workers(manager)
    await runner._drain_once()
    assert manager.events == ["evicted", "session", "registered", "cleanup"]
    assert prepared_parent[0]["actor"] == str(actor)
    assert not prepared_parent[0]["directory"].exists()
    with store.session() as db:
        run = db.get(Run, published_job.job_id)
        assert run.status == {"done": RunStatus.done, "cleanup_failure": RunStatus.failed,
                              "cancel_before_terminal": RunStatus.cancelled}[outcome]
        assert q.get_model(db, UUID(published_job.config.model_name)).definition_id == canonical_digest(published_job.definition)
        assert db.get(store.Tokenizer, published_job.definition.tokenizer.tokenizer_id) is not None
    history = registry.get(str(published_job.job_id)).history
    assert any(event.kind == "train" and event.data["step"] == 3 for event in history)
    assert any(event.kind == "checkpoint" and event.data["final"] for event in history)
    with gate.admit() as next_lease:
        next_lease.require_active()


async def test_saved_recipe_resume_retains_pins_after_history_archive_and_named_tokenizer_removal(
        client, published_job, tmp_path):
    """Signed resume inherits retained authority after history disappears; SQLite explicitly removes the production-FK cascade child."""
    from api import training_publications
    from tests.test_training_publications import granted

    checked = published_job.write(tmp_path / "output", final=True)
    notice, artifacts = granted(checked, tmp_path)
    training_publications.adopt(notice, artifacts, job_id=published_job.job_id)
    with store.session() as db:
        db.get(Run, published_job.job_id).status = RunStatus.done
        q.delete_run(db, published_job.job_id)
        db.execute(delete(RunArchitecturePin).where(RunArchitecturePin.job_id == published_job.job_id))
        root = catalog.get_resource(db, "architecture", published_job.definition.architecture.resource_id)
        root.archived = True
    assert not storage.tokenizer_build_dir(str(published_job.definition.tokenizer.tokenizer_id)).exists()
    payload = {"model_name": "continued", "resume_from": published_job.config.model_name,
               "batch_size": 2, "precision": "fp32", "compile_model": False}
    operation = uuid4()
    admin = headers(["administration"])
    response = await client.post(f"/train?request_id={operation}", headers=admin, json=payload)
    assert response.status_code == 202, response.text
    override = await client.post(f"/train?request_id={uuid4()}", headers=admin, json={**payload, "ctx_window": 8})
    assert override.status_code == 422
    with store.session() as db:
        pin = training_jobs.inherited_pin(db, UUID(response.json()["run_id"]))
        assert pin.evidence.tokenizer == published_job.definition.tokenizer
        assert pin.parameters == published_job.definition.parameters
        saved = training_handler.saved_package(TrainConfig.model_validate(response.json()["config"]), pin)
        assert saved.manifest.definition == published_job.definition
        assert db.get(RunArchitecturePin, published_job.job_id) is None
    chain = await client.post(f"/runs/chain?request_id={uuid4()}", headers=admin,
        json=[{"kind": "train", "config": payload}, {"kind": "train", "config": {}}])
    assert chain.status_code == 202, chain.text
    with store.session() as db:
        pins = [training_jobs.inherited_pin(db, UUID(value["run_id"])) for value in chain.json()]
        assert pins[0].snapshot == pins[1].snapshot == pin.snapshot
        assert pins[0].evidence.tokenizer == published_job.definition.tokenizer
    assert (await client.post(f"/train?request_id={uuid4()}", headers=headers(["public"]), json=payload)).status_code == 403
    with store.session() as db:
        q.delete_model(db, UUID(published_job.config.model_name))
    claims = jwt.decode(admin["Authorization"].removeprefix("Bearer "), SECRET, algorithms=["HS256"], options={"verify_aud": False})
    claims["scope"] = ["public"]
    public = {"Authorization": "Bearer " + jwt.encode(claims, SECRET, algorithm="HS256")}
    replay = await client.post(f"/train?request_id={operation}", headers=public, json=payload)
    assert replay.status_code == 403


@pytest.mark.parametrize("cancelled", [False, True])
async def test_parent_preparation_interruptions_preserve_cancellation_and_failure_meaning(
        published_job, prepared_parent, tmp_path, monkeypatch, cancelled):
    """Owned user cancellation becomes cancelled, while unrelated interrupted I/O remains failed; neither starts a numerical session."""
    registry = Registry()
    registry.bind_loop(asyncio.get_running_loop())
    gate = device.DeviceGate()
    monkeypatch.setattr(device, "GATE", gate)
    manager = FixtureManager(tmp_path / "runtime", published_job)

    def interrupted(cfg, **kwargs):
        """Interrupt the real parent preparation boundary with or without its owning cancellation signal."""
        if cancelled:
            kwargs["cancel"].set()
        raise InterruptedError("fixture input interruption")

    monkeypatch.setattr(training_handler.modular_data, "prepare", interrupted)
    with store.session() as db:
        db.get(Run, published_job.job_id).status = RunStatus.queued
    runner = Runner(registry, {RunKind.train: handlers.train})
    runner.attach_workers(manager)
    await runner._drain_once()
    with store.session() as db:
        assert db.get(Run, published_job.job_id).status == (RunStatus.cancelled if cancelled else RunStatus.failed)
        assert q.get_model(db, UUID(published_job.config.model_name)) is None
    assert manager.events == ["evicted"]
    assert not list(storage.package_staging_dir().iterdir())
    with gate.admit() as lease:
        lease.require_active()
