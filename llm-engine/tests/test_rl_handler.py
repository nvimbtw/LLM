"""Actual queue/actor/grant/adoption ordering around explicitly synthetic RL numerical output; live supervision is a separate gate."""

import asyncio
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest

import store
from api import handlers, recovery, rl_effects, rl_handler, training_admission, training_jobs, training_publications
from api.runner import Runner
from api.runs import Registry
from llm import backend, storage
from llm.architecture import package, package_transfer
from llm.architecture.package_integrity import decode_json, encode_json
from llm.architecture.spec import DefinitionPackage, canonical_digest
from llm.config import RlConfig
from llm.gym.effect_client import EffectClient
from llm.model import device
from llm.runtime.artifacts import ArtifactStore
from llm.runtime.protocol import ArtifactReady, Completed, EffectRequest
from llm.runtime.supervisor import CleanupUncertain
from llm.train.model_preflight import rl_profile
from llm.train.modular_records import TrainingGeneration, TrainingPublication
from llm.train.rl_records import RlPreparation
from store import architecture_queries as catalog, queries as q
from store.schema import Model, Run, RunArchitecturePin, RunKind, RunStatus
from tests.test_architecture_visibility import headers
from tests.test_bounded_hop import Body
from tests.test_rl_effect_client import public_run
from tests.test_sft_publications import rebound
from tests.test_training_publications import granted, published_job, sample, sessions
from tests.test_validation_queue import client


@pytest.fixture
def rl_job(published_job, tmp_path):
    """Register labeled synthetic base bytes and queue RL with an actual actor, reserved output and immutable inherited source pin."""
    base = published_job
    checked = base.write(tmp_path / "base", final=True)
    notice, artifacts = granted(checked, tmp_path)
    training_publications.adopt(notice, artifacts, job_id=base.job_id)
    cfg = RlConfig(model_name=str(uuid4()), base_model=base.config.model_name, task_types=["bash_ls_easy"],
        tasks_per_batch=1, group_size=2, iterations=2, max_rounds=1, checkpoint_target=2)
    identity = base.receipt.identity.model_copy(update={"profile": rl_profile(cfg, 8, "cpu"),
        "input_profile_digest": canonical_digest({"tensor_inputs": {}, "domains": [
            domain.model_dump(mode="json") for domain in rl_profile(cfg, 8, "cpu").domains]})})
    job = replace(base, job_id=uuid4(), config=cfg, receipt=base.receipt.model_copy(update={"identity": identity}))
    job = rebound(job, base.definition.tokenizer, base.files["tokenizer.json"])
    actor = uuid4()
    with store.session() as db:
        db.get(Run, base.job_id).status = RunStatus.done
        run = q.enqueue(db, id=job.job_id, kind=RunKind.rl, label="RL parent fixture", owner_id=actor,
            status=RunStatus.running, config=cfg.model_dump(mode="json", exclude_unset=True))
        catalog.attach_training_pin(db, run, catalog.prepare_saved_training_pin(db, checked.manifest.definition_id))
    return job, actor


def rl_output(job, root, *, final=False, step=1):
    """Seal complete synthetic RL publication bytes to test only parent transactions, without numerical or process claims."""
    cfg = job.config
    generation = TrainingGeneration(model_id=UUID(cfg.model_name) if final else uuid4(), index=0 if final else step,
        step=cfg.iterations if final else step, total_steps=cfg.iterations, epoch=1, stage=0, stage_name=None,
        path="model.pt" if final else "train_state.pt", architecture={"ctx_window": 8}, final=final)
    result = {"recovery": not final, "checkpoint": generation.model_dump(mode="json"),
        "definition_id": canonical_digest(job.definition), "vocab": job.definition.tokenizer.vocabulary_size,
        "ctx_window": 8, "parameters": 123, "base_model": cfg.base_model}
    metadata = TrainingPublication(job_id=job.job_id, training_kind="rl", config=cfg, generation=generation, result=result)
    files = {**job.files, "model.pt": b"synthetic RL tensor bytes", "meta.json": metadata.model_dump_json().encode()}
    if not final:
        files["train_state.pt"] = b"synthetic RL optimizer bytes"
    manifest = DefinitionPackage(format="slate.modular-model", format_version=1,
        definition_id=canonical_digest(job.definition), definition=job.definition, receipt_ids=(job.receipt.receipt_id,),
        graph_hashes=tuple(name[7:-5] for name in files if name.startswith("graphs/")), parameter_precision="fp32",
        training_state_path=None if final else "train_state.pt", optimizer_parameter_order=() if final else ("fixture.weight",))
    path = root / str(generation.model_id)
    package.publish(path, manifest, files)
    return package.verify(path)


@pytest.mark.parametrize("outcome", ["done", "cancel", "cleanup_failure", "lost_step"])
async def test_default_rl_queue_owns_actor_lease_effects_and_terminal_cleanup(rl_job, tmp_path, monkeypatch, outcome):
    """The production handler preserves actor/effect ownership and waits for the session cleanup boundary around synthetic output adoption."""
    job, actor = rl_job
    events, calls, issued = [], [], []

    def respond(request):
        """Provide bounded external gym responses while retaining the actual parent Hop, dispatcher and durable write intents."""
        calls.append(request.url.path)
        path = request.url.path
        body = decode_json(request.content) if request.content else None
        if path == "/internal/gym/runs":
            result = public_run(body["count"], seed_base=body["seed_base"], group_size=body["group_size"])
            issued.append(result)
            events.append("gym:create")
        elif path.endswith("/finish"):
            result = {"score": 1.0, "answer": "graded"}
        elif path.endswith("/step"):
            events.append("gym:step")
            raise httpx.ReadError("lost tool result")
        elif path.endswith("/close"):
            events.append("gym:close")
            result = {"episode_id": path.split("/")[-2], "status": "closed"}
        else:
            result = {"id": path.split("/")[-2], "status": body["status"]}
            events.append(f"gym:{body['status']}")
        return httpx.Response(200, stream=Body((encode_json(result),)))

    def connect(owner):
        """Require the queued actor and keep real JWT minting/stream bounds while replacing only HTTP socket transport."""
        assert owner == str(actor)
        hop = backend.Hop(owner, url="http://rust.test")
        hop._http.close()
        hop._http = httpx.Client(transport=httpx.MockTransport(respond))
        return hop

    class Session:
        """Own real artifact grants for a deliberately nonnumerical session fixture."""

        def __init__(self, root, policy):
            """Create finite private grants without pretending to launch or certify a worker."""
            self.artifacts = ArtifactStore(root / "grants", policy.resources)

        def run(self, command, *, request_id, effects, adopt, cancel):
            """Inspect actual prepared packages and drive real effects/adoption; numerical updates are explicitly omitted here."""
            prep = RlPreparation.model_validate_json(self.artifacts.read_bytes(command.config))
            assert command.training_kind == "rl" and command.corpus is None and command.pins is None
            assert prep.request == job.config and request_id == command.job_id == job.job_id
            assert effects.actor == str(actor) and effects.dispatch.log.path.parent == storage.gym_effect_dir(job.job_id)
            self.artifacts.verify(command.recovery)
            self.artifacts.verify(prep.tokenizer)

            def effect(operation):
                """Use the real parent broker and operation journal for this fixture's interpreted requests."""
                request = EffectRequest(type="effect_request", operation_id=uuid4(), effect=operation)
                return effects.begin(request, self.artifacts).result(timeout=2)

            with EffectClient(SimpleNamespace(artifacts=self.artifacts, effect=effect, check_cancelled=lambda: None)) as client:
                for iteration in range(job.config.iterations):
                    run_id, episodes = client.create_run(gym_id="bash", model_name=job.config.base_model,
                        task_types=["bash_ls_easy"], count=2, seed_base=iteration, group_size=2)
                    if outcome == "lost_step":
                        client.step(episodes[0].id, "bash", "echo once", "sampled reply")
                    for episode in episodes:
                        client.finish(episode.id, "reply")
                    client.set_status(run_id, "done")
            checked = rl_output(job, tmp_path / "synthetic-worker", final=True)
            archive = package_transfer.write_archive(checked.directory, tmp_path / "output.tar")
            notice = ArtifactReady(type="artifact_ready", publication_id=uuid4(),
                artifact=self.artifacts.stage_file(archive, "model_package_v1"), completed_step=job.config.iterations, final=True)
            adopt(notice, self.artifacts)
            events.append("registered")
            if outcome == "cancel":
                cancel.set()
            result = {"model_id": job.config.model_name, "definition_id": checked.manifest.definition_id,
                "iterations": job.config.iterations, "base_model": job.config.base_model, "ctx_window": 8,
                "vocab": job.definition.tokenizer.vocabulary_size}
            return Completed(type="completed", outcome="ok", result=self.artifacts.write_bytes(encode_json(result), "training_result_v1"))

    class Manager:
        """Present the real parent runtime interface around an explicitly synthetic cleanup transition."""

        root = tmp_path / "runtime"

        def evict_for_queue(self):
            """Record queue priority before the runner acquires the actual parent device lease."""
            events.append("evicted")

        @contextmanager
        def session(self, lease, policy, factory):
            """Retain the actual lease through registration and report cleanup uncertainty before terminal queue state."""
            lease.require_active()
            assert factory == "llm.train.rl_worker:RlBackend"
            with pytest.raises(device.Busy):
                with lease.gate.admit():
                    pytest.fail("duplicate RL device owner")
            events.append("session")
            try:
                yield Session(self.root, policy)
            finally:
                with store.session() as db:
                    assert db.get(Run, job.job_id).status == RunStatus.running
                events.append("cleanup")
                if outcome == "cleanup_failure":
                    raise CleanupUncertain("synthetic RL cleanup refusal")

    def forbidden(*args, **kwargs):
        """Ensure the parent cannot invoke legacy RL, import custom source or unpickle tensor/model state."""
        pytest.fail("parent crossed the RL numerical isolation boundary")

    Manager.root.mkdir()
    monkeypatch.setattr(device, "GATE", device.DeviceGate())
    monkeypatch.setattr(rl_effects, "Hop", connect)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(training_admission, "capacity", lambda path: training_admission.Capacity(8 << 30, 4 << 30))
    monkeypatch.setattr(handlers.rl_pipeline, "run", forbidden)
    monkeypatch.setattr("torch.load", forbidden)
    monkeypatch.setattr("llm.architecture.source.SourceLoader.load", forbidden)
    with store.session() as db:
        db.get(Run, job.job_id).status = RunStatus.queued
    registry = Registry()
    registry.bind_loop(asyncio.get_running_loop())
    runner = Runner(registry, {RunKind.rl: handlers.reinforce})
    runner.attach_workers(Manager())
    await runner._drain_once()
    with store.session() as db:
        run = db.get(Run, job.job_id)
        expected = {"done": RunStatus.done, "cancel": RunStatus.cancelled,
                    "cleanup_failure": RunStatus.failed, "lost_step": RunStatus.failed}[outcome]
        assert run.status == expected, run.error
        if outcome != "lost_step":
            assert db.get(Model, UUID(job.config.model_name)).family == "rl"
        else:
            assert db.get(Model, UUID(job.config.model_name)) is None
            assert calls.count(f"/internal/gym/episodes/{issued[0]['episodes'][0]['id']}/step") == 1
            assert events.index("cleanup") < events.index("gym:close")
    assert events[:2] == ["evicted", "session"] and "cleanup" in events
    assert storage.gym_effect_dir(job.job_id).exists() == (outcome in {"cleanup_failure", "lost_step"})
    assert not list(storage.package_staging_dir().iterdir())


async def test_signed_rl_submission_retry_and_recovery_retain_saved_pins_and_actor_identity(client, rl_job, tmp_path, monkeypatch):
    """Administration/request identity pins RL source through submit/replay/recovery and rejects structural overrides without numerical work."""
    job, _actor = rl_job
    payload = job.config.model_dump(mode="json", exclude_unset=True)
    operation = uuid4()
    admin = headers(["administration"])

    def forbidden(*args, **kwargs):
        """Detect source construction or tensor loading in signed queue-only operations."""
        pytest.fail("RL HTTP submission executed numerical work")

    monkeypatch.setattr("torch.load", forbidden)
    monkeypatch.setattr("llm.architecture.source.SourceLoader.load", forbidden)
    assert (await client.post("/rl", headers=admin, json=payload)).status_code == 422
    assert (await client.post(f"/rl?request_id={operation}", headers=headers(["public"]), json=payload)).status_code == 403
    first = await client.post(f"/rl?request_id={operation}", headers=admin, json=payload)
    assert first.status_code == 202, first.text
    same = await client.post(f"/rl?request_id={operation}", headers=admin, json=payload)
    assert same.status_code == 202 and same.json()["run_id"] == first.json()["run_id"]
    assert (await client.post(f"/rl?request_id={operation}", headers=admin, json={**payload, "lr": 0.2})).status_code == 409
    assert (await client.post(f"/rl?request_id={uuid4()}", headers=admin, json={**payload, "architecture": {}})).status_code == 422
    with store.session() as db:
        queued = db.get(Run, UUID(first.json()["run_id"]))
        assert queued.training.training_type == "rl"
        assert training_jobs.inherited_pin(db, queued.id).evidence.tokenizer == job.definition.tokenizer
        db.get(Run, job.job_id).status = RunStatus.failed
    checked = rl_output(job, tmp_path / "checkpoint")
    notice, artifacts = granted(checked, tmp_path)
    training_publications.adopt(notice, artifacts, job_id=job.job_id)
    recovered = await client.post(f"/runs/{job.job_id}/recover?request_id={uuid4()}", headers=admin)
    assert recovered.status_code == 202, recovered.text
    resumed = RlConfig.model_validate(recovered.json()["config"])
    assert resumed.base_model == job.config.base_model and resumed.resume_from == str(training_publications.metadata(checked).generation.model_id)
    with store.session() as db:
        pin = training_jobs.inherited_pin(db, UUID(recovered.json()["run_id"]))
        source, reference, step = rl_handler.source_packages(resumed, pin)
        assert source.manifest.definition == reference.manifest.definition == job.definition and step == 1
        assert db.get(RunArchitecturePin, UUID(recovered.json()["run_id"])) is not None


def test_rl_registration_failure_retains_an_adoptable_package_without_false_job_success(rl_job, tmp_path, monkeypatch):
    """Post-rename registration rollback preserves complete bytes and later adoption while the original RL job stays failed."""
    job, _actor = rl_job
    checked = rl_output(job, tmp_path / "worker")
    notice, artifacts = granted(checked, tmp_path)

    def unavailable(*args, **kwargs):
        """Interrupt the real registration transaction after package rename and evidence writes."""
        raise RuntimeError("RL model index unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(q, "upsert_model", unavailable)
        with pytest.raises(RuntimeError, match="model index unavailable"):
            training_publications.adopt(notice, artifacts, job_id=job.job_id)
    model_id = training_publications.metadata(checked).generation.model_id
    assert storage.model_dir(str(model_id)).exists()
    with store.session() as db:
        assert db.get(Model, model_id) is None
        db.get(Run, job.job_id).status = RunStatus.failed
    recovery.reconcile()
    with store.session() as db:
        model = db.get(Model, model_id)
        assert model.family == "rl" and model.definition_id == canonical_digest(job.definition)
        assert db.get(Run, job.job_id).status == RunStatus.failed
