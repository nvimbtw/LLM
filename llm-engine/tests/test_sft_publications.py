"""Parent SFT authority/adoption and queue transitions with explicitly synthetic numerical outputs."""

import asyncio
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Event
from uuid import UUID, uuid4

import numpy as np
import pytest

import store
from api import chain, handlers, recovery, sft_authority, training_admission, training_jobs, training_publications
from api.runner import Runner
from api.runs import Registry
from llm import storage
from llm.architecture import package, package_transfer
from llm.architecture.graphs import GraphArtifacts, export_guards
from llm.architecture.package_integrity import encode_json
from llm.architecture.spec import canonical_digest
from llm.architecture.tokenizer_binding import binding_from_bytes
from llm.config import SftConfig
from llm.corpus import remote, pack
from llm.corpus.spans import DocSpan, Spans
from llm.chat.template import Reasoning
from llm.model import device
from llm.orchestrate import modular_sft_data
from llm.orchestrate.modular_data import PreparedData
from llm.runtime.artifacts import ArtifactStore
from llm.runtime.protocol import ArtifactReady, Completed
from llm.runtime.supervisor import CleanupUncertain
from llm.train.data_identity import fingerprint
from llm.train.modular_records import SftDataset, TrainingGeneration, TrainingPreparation, TrainingPublication
from llm.train.plan import DataPlan
from llm.train.sft_tokenizer import prepare_tokenizer
from store import architecture_queries as catalog, queries as q
from store.schema import Model, Run, RunArchitecturePin, RunKind, RunStatus
from tests.test_sft_tokenizer import legacy_bytes
from tests.test_training_publications import granted, published_job, sample, sessions
from tests.test_validation_queue import client
from tests.test_architecture_visibility import headers


def rebound(job, binding, raw):
    """Rebuild labeled synthetic graph/receipt identity for actual tokenizer bytes without claiming numerical validation."""
    definition = job.definition.model_copy(update={"tokenizer": binding})
    identity = job.receipt.identity.model_copy(update={"tokenizer": binding})
    graphs = GraphArtifacts("")
    for mode in identity.profile.modes:
        graphs.add(view="operations", mode=mode, signature="full_prefix", scope="root", strict=True, guards=(),
            nodes=({"id": "fixture", "kind": "input", "label": "Synthetic SFT transaction fixture", "occurrence_path": "fixture"},), edges=())
    identity = identity.model_copy(update={"profile": identity.profile.model_copy(update={"export_guards": export_guards(graphs.entries)})})
    graphs.identity_hash = canonical_digest(identity)
    graph_hash, blobs = graphs.finish()
    receipt = job.receipt.model_copy(update={"receipt_id": uuid4(), "identity": identity,
        "identity_hash": canonical_digest(identity), "graph_manifest_hash": graph_hash})
    files = {name: value for name, value in job.files.items() if not name.startswith(("graphs/", "receipts/"))}
    files.update({"tokenizer.json": raw, f"receipts/{receipt.receipt_id}.json": encode_json(receipt)})
    files.update({f"graphs/{digest}.json": source.encode() for digest, source in blobs.items()})
    return replace(job, definition=definition, receipt=receipt, files=files)


@pytest.fixture
def sft_job(published_job, tmp_path):
    """Register an actual immutable base package and queue SFT with its retained exact source and legacy tokenizer pin."""
    raw = legacy_bytes()
    base = rebound(published_job, binding_from_bytes(uuid4(), raw), raw)
    checked = base.write(tmp_path / "base", final=True)
    notice, artifacts = granted(checked, tmp_path)
    training_publications.adopt(notice, artifacts, job_id=base.job_id)
    cfg = SftConfig(model_name=str(uuid4()), base_model=base.config.model_name,
        datasets=["fixture"], epochs=1, batch_size=2, precision="fp32", compile_model=False)
    job_id = uuid4()
    with store.session() as db:
        db.get(Run, base.job_id).status = RunStatus.done
        run = q.enqueue(db, id=job_id, kind=RunKind.train, label="SFT parent fixture", status=RunStatus.running,
            config=cfg.model_dump(mode="json", exclude_unset=True))
        pin = catalog.prepare_saved_training_pin(db, checked.manifest.definition_id)
        catalog.attach_training_pin(db, run, pin)
    return replace(base, job_id=job_id, config=cfg), package.verify(storage.model_dir(cfg.base_model))


def prepared_data(job, directory):
    """Write real aligned corpus/mask/span/tokenizer files with an allowed private memory-label derivation."""
    directory.mkdir(parents=True, exist_ok=True)
    raw, binding = prepare_tokenizer(job.files["tokenizer.json"], job.definition.tokenizer, [47, 48])
    corpus = np.arange(49, dtype=np.uint16)
    mask = np.ones(49, dtype=np.uint8)
    mask[:16] = 0
    documents = Spans.of([DocSpan("sft", 0, 49)])
    corpus.tofile(directory / "corpus.bin")
    mask.tofile(directory / "mask.bin")
    documents.save(directory / "spans.npy")
    (directory / "tokenizer.json").write_bytes(raw)
    cfg = job.config
    plan = DataPlan(documents=documents, eot_id=0, shuffle=cfg.shuffle, seed=cfg.seed)
    data = PreparedData(directory / "corpus.bin", directory / "spans.npy", directory / "tokenizer.json",
        binding, 49, ("sft",), (), 0, "sft-fixture", fingerprint(corpus, mask, plan=plan), None, cfg.shuffle, cfg.seed)
    dataset = SftDataset(sets=(), reasoning="keep", conversations=1, total_tokens=49,
        trainable_tokens=33, skipped=0, special_ids_seen=(47, 48))
    return modular_sft_data.PreparedSftData(data, directory / "mask.bin", dataset)


def sft_output(job, prepared, directory, *, final=False, result_changes=None):
    """Seal a complete synthetic SFT generation whose provenance can be checked independently by the real adopter."""
    rebound_job = rebound(job, prepared.data.binding, prepared.data.tokenizer.read_bytes())
    generation = TrainingGeneration(model_id=UUID(job.config.model_name) if final else uuid4(), index=0 if final else 1,
        step=3 if final else 1, total_steps=3, epoch=1, stage=0, stage_name=None,
        path="model.pt" if final else "train_state.pt", architecture={"ctx_window": 8}, final=final)
    authority = sft_authority.read(job.job_id)
    result = {"recovery": not final, "checkpoint": generation.model_dump(mode="json"),
        "definition_id": canonical_digest(rebound_job.definition), "vocab": prepared.data.binding.vocabulary_size,
        "ctx_window": 8, "parameters": 123, "base_model": job.config.base_model,
        "data_fingerprint": prepared.data.fingerprint, "dataset": authority.dataset.model_dump(mode="json"),
        "warm_start": {"status": "initialized", "initialized_ids": [], "skipped_ids": []}, "warm_started": 0,
        **(result_changes or {})}
    value = TrainingPublication(job_id=job.job_id, training_kind="sft", config=job.config, generation=generation, result=result)
    files = {**rebound_job.files, "model.pt": b"synthetic tensors", "meta.json": value.model_dump_json().encode()}
    if not final:
        files["train_state.pt"] = b"synthetic optimizer"
    from llm.architecture.spec import DefinitionPackage
    manifest = DefinitionPackage(format="slate.modular-model", format_version=1,
        definition_id=canonical_digest(rebound_job.definition), definition=rebound_job.definition,
        receipt_ids=(rebound_job.receipt.receipt_id,), graph_hashes=tuple(name[7:-5] for name in files if name.startswith("graphs/")),
        parameter_precision="fp32", training_state_path=None if final else "train_state.pt",
        optimizer_parameter_order=() if final else ("fixture.weight",))
    path = directory / str(generation.model_id)
    package.publish(path, manifest, files)
    return package.verify(path)


def authorize(job, saved, prepared):
    """Persist exactly the actual parent-prepared proof before any synthetic numerical publication."""
    authority = sft_authority.prepare(job.job_id, job.config, saved, prepared)
    sft_authority.publish(authority)
    assert sft_authority.read(job.job_id) == authority
    return authority


def test_parent_prepares_real_private_sft_data_with_actor_masks_and_derived_identity(sft_job, tmp_path, monkeypatch):
    """Actual canonical rendering hashes corpus/mask together and keeps different reasoning/tokenizer preparations private."""
    job, saved = sft_job
    actor = str(uuid4())
    rows = [{"role": "memory", "content": json.dumps({"memory": "the source fact", "thread": str(uuid4()),
        "message": str(uuid4()), "role": "assistant", "age_seconds": 1})},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "<|thinking|>reason<|/thinking|>answer"}]

    class Pages:
        """Supply trusted finite source rows at the service transport seam while using the actual renderer."""

        def __init__(self, initiating_actor):
            """Verify parent acquisition retains the initiating actor."""
            assert initiating_actor == actor

        def __enter__(self):
            """Open one scoped fixture source connection."""
            return self

        def __exit__(self, *args):
            """Finish source acquisition before model admission."""
            pass

        def dataset(self, identity):
            """Return versioned dataset facts for deterministic private preparation."""
            return remote.Dataset(identity, "chat fixture", "sft", 1, 128, "fixed")

        def documents(self, identity, *, cap=None):
            """Yield real structured conversation text in the backend data contract."""
            yield {"messages": rows}

    monkeypatch.setattr(remote, "Client", Pages)
    outputs = []
    for strip in (False, True):
        cfg = job.config.model_copy(update={"strip_thinking": strip})
        directory = tmp_path / f"prepared-{strip}"
        directory.mkdir()
        value = modular_sft_data.prepare(cfg, saved=saved, base=saved, directory=directory, actor=actor,
            progress=lambda _: None, cancel=Event(), budget=pack.PackBudget(max_corpus_bytes=1 << 20, max_documents=1))
        corpus = np.fromfile(value.data.corpus, dtype=np.uint16)
        mask = np.fromfile(value.mask, dtype=np.uint8)
        documents = Spans.load(value.data.spans, value.data.names)
        plan = DataPlan(documents=documents, eot_id=value.data.eot_id, shuffle=cfg.shuffle, seed=cfg.seed)
        assert value.data.fingerprint == fingerprint(corpus, mask, plan=plan)
        assert value.dataset.trainable_tokens == int(mask.sum()) > 0
        assert value.data.binding.derived_from == saved.manifest.definition.tokenizer.fingerprint
        assert {47, 48}.issubset(value.dataset.special_ids_seen)
        assert value.dataset.reasoning == ("strip" if strip else "keep")
        assert value.chat_probe is None and value.pretrain_probe is None
        outputs.append(value)
    assert outputs[0].data.key != outputs[1].data.key
    assert outputs[0].data.fingerprint != outputs[1].data.fingerprint
    assert (saved.directory / "tokenizer.json").read_bytes() == job.files["tokenizer.json"]


@pytest.mark.parametrize("row", [None, [], {"system": 1, "turns": []}, {"turns": [1]},
    {"turns": [{"role": 1}]}, {"turns": [{"role": "assistant", "text": []}]},
    {"turns": [{"role": "assistant", "tool": []}]}, {"turns": [{"role": "assistant", "argument": {}}]}])
def test_malformed_optional_probe_rows_cannot_reach_the_renderer(row):
    """Invalid optional held-out data is omitted at its interpreted boundary without failing numerical training."""
    assert modular_sft_data.probe_turns(row) is None


def test_sft_adoption_keeps_enqueue_pin_and_retains_derived_definition(sft_job, tmp_path, monkeypatch):
    """Only durable parent-prepared actual bytes authorize tokenizer adoption; source imports and tensor loads stay outside the parent."""
    job, saved = sft_job
    prepared = prepared_data(job, tmp_path / "data")
    authority = authorize(job, saved, prepared)
    sft_authority.publish(authority)

    def forbidden(*args, **kwargs):
        """Detect parent execution of source or tensor loading during publication."""
        pytest.fail("parent executed numerical source")

    monkeypatch.setattr("torch.load", forbidden)
    monkeypatch.setattr("llm.architecture.source.SourceLoader.load", forbidden)
    checked = sft_output(job, prepared, tmp_path / "worker", final=True)
    notice, artifacts = granted(checked, tmp_path)
    frame = training_publications.adopt(notice, artifacts, job_id=job.job_id)
    assert training_publications.adopt(notice, artifacts, job_id=job.job_id) == frame
    with store.session() as db:
        pin = training_jobs.inherited_pin(db, job.job_id)
        assert pin.evidence.tokenizer == saved.manifest.definition.tokenizer
        model = db.get(Model, UUID(frame.model_id))
        assert model.family == "sft" and model.definition_id == checked.manifest.definition_id
        assert model.parameters["result"]["dataset"]["marker_history_complete"] is False
    assert prepared.data.binding != pin.evidence.tokenizer
    assert (saved.directory / "tokenizer.json").read_bytes() == job.files["tokenizer.json"]


@pytest.mark.parametrize("mismatch", ["missing", "fingerprint", "dataset", "rows", "count", "source", "config"])
def test_sft_parent_refuses_missing_or_forged_authority_before_rename(sft_job, tmp_path, mismatch):
    """Worker provenance alone cannot alter prepared data, source, row initialization or original configuration."""
    job, saved = sft_job
    prepared = prepared_data(job, tmp_path / "data")
    authority = authorize(job, saved, prepared)
    changes = {"fingerprint": {"data_fingerprint": "a" * 64}, "dataset": {"dataset": prepared.dataset.model_dump(mode="json")},
        "rows": {"warm_start": {"status": "initialized", "initialized_ids": [47]}, "warm_started": 1},
        "count": {"warm_started": 1}}
    checked = sft_output(job, prepared, tmp_path / "worker", result_changes=changes.get(mismatch))
    if mismatch == "missing":
        sft_authority.discard(job.job_id)
    elif mismatch in {"source", "config"}:
        changed = authority.model_copy(update={"source_model_id": uuid4()} if mismatch == "source" else {"config_digest": "f" * 64})
        (storage.training_preparation_dir(job.job_id) / "prepared.json").write_bytes(changed.model_dump_json().encode())
    notice, artifacts = granted(checked, tmp_path)
    with pytest.raises((OSError, ValueError)):
        training_publications.adopt(notice, artifacts, job_id=job.job_id)
    assert not storage.model_dir(checked.directory.name).exists()
    with store.session() as db:
        assert catalog.evidence_for_job(db, job.job_id) is None


def test_sft_registration_failure_preserves_authority_for_startup(sft_job, tmp_path, monkeypatch):
    """A crash after immutable rename keeps proof and bytes for real startup adoption without changing the failed job outcome."""
    job, saved = sft_job
    prepared = prepared_data(job, tmp_path / "data")
    authorize(job, saved, prepared)
    checked = sft_output(job, prepared, tmp_path / "worker")
    notice, artifacts = granted(checked, tmp_path)

    def failed_index(*args, **kwargs):
        """Interrupt the real database transaction after the complete output directory exists."""
        raise RuntimeError("SFT model index unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(q, "upsert_model", failed_index)
        with pytest.raises(RuntimeError, match="index unavailable"):
            training_publications.adopt(notice, artifacts, job_id=job.job_id)
    assert sft_authority.read(job.job_id).tokenizer == prepared.data.binding
    with store.session() as db:
        db.get(Run, job.job_id).status = RunStatus.failed
    recovery.reconcile()
    with store.session() as db:
        assert db.get(Model, UUID(checked.directory.name)).definition_id == checked.manifest.definition_id
        assert db.get(Run, job.job_id).status == RunStatus.failed
    assert not storage.training_preparation_dir(job.job_id).exists()


def test_sft_authority_cleanup_keeps_unadopted_or_active_inputs(sft_job, tmp_path):
    """Startup cleanup preserves an unresolved package's authority, then retires terminal/pruned proof only after adoption."""
    job, saved = sft_job
    prepared = prepared_data(job, tmp_path / "data")
    authority = authorize(job, saved, prepared)
    with pytest.raises(ValueError, match="different prepared-input"):
        sft_authority.publish(authority.model_copy(update={"data_fingerprint": "a" * 64}))
    assert sft_authority.read(job.job_id) == authority
    sft_authority.reconcile()
    assert storage.training_preparation_dir(job.job_id).exists()
    checked = sft_output(job, prepared, tmp_path / "worker")
    archive = package_transfer.write_archive(checked.directory, tmp_path / "pending.tar")
    destination = storage.model_dir(checked.directory.name)
    package_transfer.materialize(archive, destination)
    with store.session() as db:
        db.get(Run, job.job_id).status = RunStatus.failed
    sft_authority.reconcile()
    assert storage.training_preparation_dir(job.job_id).exists()
    training_publications.reconcile(destination)
    with store.session() as db:
        q.delete_run(db, job.job_id)
    sft_authority.reconcile()
    assert not storage.training_preparation_dir(job.job_id).exists()
    assert training_publications.register(package.verify(destination)).model_id == destination.name


@pytest.mark.parametrize("outcome", ["done", "cleanup_failure", "cancel", "derived_recovery", "pending_chain"])
async def test_sft_queue_owns_one_lease_and_finishes_after_cleanup(client, sft_job, tmp_path, monkeypatch, outcome):
    """The actual runner selects supervised SFT and preserves completion/cancellation/cleanup ordering around synthetic outputs."""
    job, saved = sft_job
    if outcome == "pending_chain":
        source = TrainingPublication.model_validate_json((saved.directory / "meta.json").read_bytes())
        cfg = job.config.model_copy(update={"model_name": str(uuid4())})
        with store.session() as db:
            inherited = training_jobs.inherited_pin(db, source.job_id)
            assert inherited.evidence.tokenizer.kind == "pending"
            run = q.enqueue(db, id=uuid4(), kind=RunKind.train, label="waiting SFT chain",
                config=cfg.model_dump(mode="json", exclude={"base_model"}, exclude_unset=True),
                status=RunStatus.waiting, depends_on=source.job_id)
            catalog.attach_training_pin(db, run, inherited)
            next_id = run.id
        assert chain.promote(source.job_id, UUID(cfg.base_model)) == 1
        with store.session() as db:
            assert q.delete_run(db, source.job_id)
            catalog.get_resource(db, "architecture", job.definition.architecture.resource_id).archived = True
        job = replace(job, job_id=next_id, config=cfg)
    if outcome == "derived_recovery":
        first = prepared_data(job, tmp_path / "original-data")
        authorize(job, saved, first)
        checked = sft_output(job, first, tmp_path / "original-worker")
        notice, artifacts = granted(checked, tmp_path)
        training_publications.adopt(notice, artifacts, job_id=job.job_id)
        with store.session() as db:
            db.get(Run, job.job_id).status = RunStatus.failed
        response = await client.post(f"/runs/{job.job_id}/recover?request_id={uuid4()}", headers=headers(["administration"]))
        assert response.status_code == 202, response.text
        job = replace(job, job_id=UUID(response.json()["run_id"]), config=SftConfig.model_validate(response.json()["config"]))
    events = []
    actor = uuid4()

    def prepare(cfg, **kwargs):
        """Write real aligned input files at the acquisition seam and check the original initiating actor."""
        assert cfg == job.config and kwargs["actor"] == str(actor)
        return prepared_data(job, kwargs["directory"])

    class Session:
        """Own actual grants for the synthetic numerical half of the complete queue test."""

        def __init__(self, root, policy):
            """Create bounded private grant storage under this fixture's session owner."""
            self.artifacts = ArtifactStore(root / "grants", policy.resources)

        def run(self, command, *, request_id, adopt, cancel):
            """Validate actual SFT/private grants and use the real adopter before reporting synthetic completion."""
            prep = TrainingPreparation.model_validate_json(self.artifacts.read_bytes(command.config))
            assert prep.training_kind == command.training_kind == "sft"
            assert prep.sft.request == job.config and request_id == job.job_id
            assert command.recovery.format == "model_package_v1"
            assert self.artifacts.read_bytes(prep.sft.mask) == bytes([0] * 16 + [1] * 33)
            data = prepared_data(job, tmp_path / "output-inputs")
            checked = sft_output(job, data, tmp_path / "worker", final=True)
            archive = package_transfer.write_archive(checked.directory, tmp_path / "output.tar")
            notice = ArtifactReady(type="artifact_ready", publication_id=uuid4(),
                artifact=self.artifacts.stage_file(archive, "model_package_v1"), completed_step=3, final=True)
            adopt(notice, self.artifacts)
            events.append("registered")
            if outcome == "cancel":
                cancel.set()
            result = {"model_id": job.config.model_name, "definition_id": checked.manifest.definition_id,
                "steps": 3, "total_steps": 3, "data_fingerprint": prep.data.fingerprint}
            return Completed(type="completed", outcome="ok", result=self.artifacts.write_bytes(encode_json(result), "training_result_v1"))

    class Manager:
        """Expose the production runtime interface without fabricating a process cleanup certificate."""

        root = tmp_path / "runtime"

        def evict_for_queue(self):
            """Record the runner's queue priority boundary."""
            events.append("evicted")

        @contextmanager
        def session(self, lease, policy, factory):
            """Retain the actual device lease through registration and a possibly failing cleanup boundary."""
            lease.require_active()
            assert factory == "llm.train.modular_worker:TrainingBackend"
            with pytest.raises(device.Busy):
                with lease.gate.admit():
                    pytest.fail("duplicate device owner")
            events.append("session")
            try:
                yield Session(self.root, policy)
            finally:
                with store.session() as db:
                    assert db.get(Run, job.job_id).status == RunStatus.running
                events.append("cleanup")
                if outcome == "cleanup_failure":
                    raise CleanupUncertain("SFT synthetic cleanup refusal")

    registry = Registry()
    registry.bind_loop(asyncio.get_running_loop())
    monkeypatch.setattr(device, "GATE", device.DeviceGate())
    monkeypatch.setattr(modular_sft_data, "prepare", prepare)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(training_admission, "capacity", lambda path: training_admission.Capacity(8 << 30, 4 << 30))
    with store.session() as db:
        run = db.get(Run, job.job_id)
        run.status, run.owner_id = RunStatus.queued, actor
    runner = Runner(registry, {RunKind.train: handlers.train})
    runner.attach_workers(Manager())
    await runner._drain_once()
    with store.session() as db:
        assert events == ["evicted", "session", "registered", "cleanup"], db.get(Run, job.job_id).error
        assert db.get(Run, job.job_id).status == {"done": RunStatus.done, "cancel": RunStatus.cancelled,
            "cleanup_failure": RunStatus.failed, "derived_recovery": RunStatus.done, "pending_chain": RunStatus.done}[outcome]
        assert db.get(Model, UUID(job.config.model_name)).family == "sft"
        pin = training_jobs.inherited_pin(db, job.job_id)
        if outcome == "pending_chain":
            assert pin.evidence == inherited.evidence and pin.snapshot == inherited.snapshot
        else:
            assert pin.evidence.tokenizer == saved.manifest.definition.tokenizer
    assert storage.training_preparation_dir(job.job_id).exists() == (outcome in {"cancel", "cleanup_failure"})
    assert not list(storage.package_staging_dir().iterdir())


async def test_signed_sft_submission_retains_saved_pins_and_operation_identity(client, sft_job, monkeypatch):
    """Modular SFT requires verified administration and a request UUID; replay retains the frozen saved definition."""
    job, saved = sft_job
    payload = job.config.model_dump(mode="json", exclude_unset=True)
    operation = uuid4()
    admin = headers(["administration"])

    def forbidden(*args, **kwargs):
        """Ensure HTTP submission never opens a corpus client or imports numerical source."""
        pytest.fail("SFT submission executed numerical/data work")

    monkeypatch.setattr(remote, "Client", forbidden)
    monkeypatch.setattr("llm.architecture.source.SourceLoader.load", forbidden)
    assert (await client.post("/sft", headers=admin, json=payload)).status_code == 422
    assert (await client.post(f"/sft?request_id={operation}", headers=headers(["public"]), json=payload)).status_code == 403
    first = await client.post(f"/sft?request_id={operation}", headers=admin, json=payload)
    assert first.status_code == 202, first.text
    same = await client.post(f"/sft?request_id={operation}", headers=admin, json=payload)
    assert same.status_code == 202 and same.json()["run_id"] == first.json()["run_id"]
    changed = await client.post(f"/sft?request_id={operation}", headers=admin, json={**payload, "lr": 0.2})
    assert changed.status_code == 409
    with store.session() as db:
        pin = training_jobs.inherited_pin(db, UUID(first.json()["run_id"]))
        assert pin.evidence.tokenizer == saved.manifest.definition.tokenizer
        assert db.get(Run, UUID(first.json()["run_id"])).training.training_type == "sft"
