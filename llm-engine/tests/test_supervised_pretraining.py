"""Runnable genuine queue-to-parent-to-child pretraining; corpus transport alone uses finite trusted fixture pages."""

import asyncio
import shutil
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
import torch

import store
from api import builtin_catalog, builtin_tokenizer, handlers, training_handler, training_jobs
from api.runner import Runner
from api.runs import Registry
from llm import storage
from llm.architecture.builtins.catalog import builtin_definitions
from llm.architecture.package import verify
from llm.corpus import remote
from llm.config import TrainConfig
from llm.runtime.supervisor import WorkerManager
from store import architecture_queries as catalog, queries as q
from store.schema import ArchitectureRevision, Run, RunKind, RunStatus
from tests.test_validation_queue import publish_fixture
from tests.test_validation_queue import client
from tests.test_architecture_visibility import headers


class CorpusPages:
    """Supply finite actual prose to the real parent packer without claiming a live Rust/data-engine transport."""

    def __init__(self, actor):
        """Retain the initiating actual caller passed to parent acquisition."""
        self.actor = actor

    def __enter__(self):
        """Expose one scoped trusted corpus fixture connection."""
        return self

    def __exit__(self, *args):
        """Close this fixture connection before the numerical worker starts."""
        pass

    def dataset(self, identity):
        """Return stable count/version facts for the explicitly requested fixture dataset."""
        return remote.Dataset(identity, "fixture prose", "pretrain", 4, 88, "fixed-version")

    def documents(self, identity, *, cap=None):
        """Yield real text for byte-level tokenization and packing, not an already forged corpus grant."""
        yield from ({"content": "one two three four five"} for _ in range(cap or 4))


def accepted_standard() -> ArchitectureRevision:
    """Publish actual trusted CPU numerical fixtures for one supplied block and its recipe, without simulating a supervised validation process."""
    builtin_catalog.seed_blocks()
    reference = builtin_definitions()["standard_transformer"].reference
    with store.session() as db:
        run = next(row for row in db.scalars(select(Run))
                   if row.config.get("builtin_root", {}).get("revision_id") == str(reference.revision_id))
        block_job = run.id
    publish_fixture(str(block_job))
    builtin_catalog.seed_recipes()
    with store.session() as db:
        run = next(row for row in db.scalars(select(Run)) if row.config.get("recipe") is not None)
        recipe_job = run.id
    publish_fixture(str(recipe_job))
    with store.session() as db:
        return catalog.revision_rows(db, "architecture", builtin_catalog.recipe_resource_id("standard"))[0]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
async def test_default_pretrain_handler_trains_and_registers_under_real_supervision(store_session, tmp_path, monkeypatch, device):
    """Exercise the actual queue/lease/guardian/training/adoption/cleanup path; CPU socket denial is a failure, not a skip."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("user-deferred CUDA hardware verification")
    if device == "cpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(remote, "Client", CorpusPages)
    revision = accepted_standard()
    tokenizer_id, _raw = builtin_tokenizer.baseline_bytes()
    cfg = TrainConfig(model_name=str(uuid4()), batch_size=2, precision="fp32" if device == "cpu" else "bf16",
        compile_model=False, checkpoint_target=2, warmup_steps=0, tokenizer_id=str(tokenizer_id), datasets=[str(uuid4())],
        architecture={"revision_id": revision.id, "parameters": {"dimensions": 8, "heads": 2, "ff_multiplier": 2, "context_window": 8}})
    config = cfg.model_dump(mode="json", exclude_unset=True)
    job_id, actor = uuid4(), uuid4()
    with store.session() as db:
        pin = training_jobs.prepare_selection(db, config, job_id)
        run = q.enqueue(db, id=job_id, kind=RunKind.train, label="supervised pretraining", status=RunStatus.running,
                        config=config, owner_id=actor)
        catalog.attach_training_pin(db, run, pin)
    registry = Registry()
    registry.bind_loop(asyncio.get_running_loop())
    with WorkerManager(tmp_path / "runtime") as manager:
        runner = Runner(registry, {RunKind.train: handlers.train})
        runner.attach_workers(manager)
        await runner._execute(job_id, RunKind.train, config, str(actor))
        with store.session() as db:
            row = db.get(Run, job_id)
            assert row.status == RunStatus.done, row.error
            assert row.training.definition_id is not None
            assert q.get_model(db, row.ref_id).definition_id == row.training.definition_id
            assert q.checkpoints_of(db, row.ref_id)
        checked = verify(storage.model_dir(cfg.model_name))
        assert checked.manifest.parameter_precision == "fp32"
        assert not list(manager.root.glob("*/ownership.json"))


async def test_actual_saved_recovery_matches_uninterrupted_cpu_weights_without_the_original_tokenizer_build(
        client, store_session, tmp_path, monkeypatch):
    """Use real workers, cancellation after durable checkpoint ACK, signed recovery and exact final weight-byte equality."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(remote, "Client", CorpusPages)
    revision = accepted_standard()
    tokenizer_id, _raw = builtin_tokenizer.baseline_bytes()
    cfg = TrainConfig(model_name=str(uuid4()), batch_size=2, seed=19, precision="fp32", compile_model=False,
        checkpoint_target=8, warmup_steps=0, tokenizer_id=str(tokenizer_id), datasets=[str(uuid4())],
        architecture={"revision_id": revision.id, "parameters": {"dimensions": 8, "heads": 2, "ff_multiplier": 2, "context_window": 8}})
    actor = uuid4()
    registry = Registry()
    registry.bind_loop(asyncio.get_running_loop())
    with WorkerManager(tmp_path / "runtime") as manager:
        runner = Runner(registry, {RunKind.train: handlers.train})
        runner.attach_workers(manager)
        ids = []
        for attempt in range(2):
            config = cfg.model_copy(update={"model_name": str(uuid4())}).model_dump(mode="json", exclude_unset=True)
            job_id = uuid4()
            with store.session() as db:
                prepared = training_jobs.prepare_selection(db, config, job_id)
                run = q.enqueue(db, id=job_id, kind=RunKind.train, status=RunStatus.running,
                                label="actual recovery", config=config, owner_id=actor)
                catalog.attach_training_pin(db, run, prepared)
            if attempt:
                original = training_handler.TrainingPublications.__call__

                def cancel_after_adoption(owner, notice, artifacts):
                    """Inject actual observer cancellation only after the real parent has committed the first complete generation."""
                    original(owner, notice, artifacts)
                    if notice.artifact.format == "model_package_v1" and not notice.final:
                        owner.context.cancel.set()

                with monkeypatch.context() as patch:
                    patch.setattr(training_handler.TrainingPublications, "__call__", cancel_after_adoption)
                    await runner._execute(job_id, RunKind.train, config, str(actor))
            else:
                await runner._execute(job_id, RunKind.train, config, str(actor))
            with store.session() as db:
                row = db.get(Run, job_id)
                assert row.status == (RunStatus.cancelled if attempt else RunStatus.done), row.error
            ids.append((job_id, config["model_name"]))
        shutil.rmtree(storage.tokenizer_build_dir(str(tokenizer_id)))
        response = await client.post(f"/runs/{ids[1][0]}/recover?request_id={uuid4()}", headers=headers(["administration"]))
        assert response.status_code == 202, response.text
        recovered_id = UUID(response.json()["run_id"])
        with store.session() as db:
            row = db.get(Run, recovered_id)
            row.status = RunStatus.running
            recovered_config, recovered_actor = dict(row.config), str(row.owner_id)
        await runner._execute(recovered_id, RunKind.train, recovered_config, recovered_actor)
        with store.session() as db:
            row = db.get(Run, recovered_id)
            assert row.status == RunStatus.done, row.error
        expected = verify(storage.model_dir(ids[0][1]))
        recovered = verify(storage.model_dir(recovered_config["model_name"]))
        assert recovered.manifest.definition == expected.manifest.definition
        assert recovered.integrity.files["model.pt"].sha256 == expected.integrity.files["model.pt"].sha256
        assert not list(manager.root.glob("*/ownership.json"))
