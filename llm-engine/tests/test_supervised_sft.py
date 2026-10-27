"""Runnable real queue/guardian SFT and signed recovery acceptance; transport pages alone use trusted fixtures."""

import asyncio
import json
from uuid import UUID, uuid4

import pytest
import torch

import store
from api import builtin_tokenizer, handlers, training_handler, training_jobs
from api.runner import Runner
from api.runs import Registry
from llm import storage
from llm.architecture.package import verify
from llm.config import SftConfig, TrainConfig
from llm.corpus import remote
from llm.runtime.supervisor import WorkerManager
from store import architecture_queries as catalog, queries as q
from store.schema import Run, RunKind, RunStatus
from tests.test_architecture_visibility import headers
from tests.test_supervised_pretraining import CorpusPages, accepted_standard
from tests.test_validation_queue import client


class ChatPages(CorpusPages):
    """Supply actual chat text at the backend transport seam while rendering/masking execute through production code."""

    def dataset(self, identity):
        """Return immutable fixture chat selection facts."""
        return remote.Dataset(identity, "SFT chat fixture", "sft", 4, 256, "fixed-chat-version")

    def documents(self, identity, *, cap=None):
        """Yield structured assistant targets and prompt-only user tokens for canonical SFT rendering."""
        yield from ({"messages": [{"role": "user", "content": "one two three four"},
            {"role": "assistant", "content": "five four three two one"}]} for _ in range(cap or 4))


def enqueue_base(cfg: TrainConfig, actor: UUID) -> UUID:
    """Freeze the actual accepted recipe and input selection in the production queue transaction."""
    job_id = uuid4()
    config = cfg.model_dump(mode="json", exclude_unset=True)
    with store.session() as db:
        pin = training_jobs.prepare_selection(db, config, job_id)
        run = q.enqueue(db, id=job_id, kind=RunKind.train, label="SFT acceptance base", status=RunStatus.running,
            config=config, owner_id=actor)
        catalog.attach_training_pin(db, run, pin)
    return job_id


async def execute_existing(runner: Runner, job_id: UUID) -> dict:
    """Run one durable queue row under the actual admitted supervisor and require registered completion after cleanup."""
    with store.session() as db:
        run = db.get(Run, job_id)
        run.status = RunStatus.running
        config, actor = dict(run.config), str(run.owner_id)
    await runner._execute(job_id, RunKind.train, config, actor)
    with store.session() as db:
        run = db.get(Run, job_id)
        assert run.status == RunStatus.done, run.error
        assert q.get_model(db, run.ref_id).definition_id == run.training.definition_id
    return config


@pytest.mark.parametrize("device", ["cpu", "cuda"])
async def test_supervised_sft_trains_reference_and_recovers_exact_saved_cursor(client, store_session, tmp_path, monkeypatch, device):
    """Exercise real pretrain/SFT workers, acknowledged cancellation, original frozen KL reference and signed recovery."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("user-deferred CUDA hardware verification")
    if device == "cpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(remote, "Client", CorpusPages)
    revision = accepted_standard()
    tokenizer_id, _raw = builtin_tokenizer.baseline_bytes()
    cfg = TrainConfig(model_name=str(uuid4()), batch_size=2, warmup_steps=0, precision="fp32" if device == "cpu" else "bf16",
        compile_model=False, checkpoint_target=0, tokenizer_id=str(tokenizer_id), datasets=[str(uuid4())],
        architecture={"revision_id": revision.id, "parameters": {"dimensions": 8, "heads": 2, "ff_multiplier": 2, "context_window": 8}})
    actor = uuid4()
    base_job = enqueue_base(cfg, actor)
    registry = Registry()
    registry.bind_loop(asyncio.get_running_loop())
    with WorkerManager(tmp_path / "runtime") as manager:
        runner = Runner(registry, {RunKind.train: handlers.train})
        runner.attach_workers(manager)
        await execute_existing(runner, base_job)
        base = verify(storage.model_dir(cfg.model_name))
        base_tokenizer = (base.directory / "tokenizer.json").read_bytes()
        monkeypatch.setattr(remote, "Client", ChatPages)
        probe = storage.eval_dir() / "chat"
        probe.mkdir(parents=True)
        (probe / "heldout.jsonl").write_text(json.dumps({"turns": [{"role": "user", "text": "one two"},
            {"role": "assistant", "text": "three four five"}]}) + "\n")
        sft = SftConfig(model_name="SFT supervised acceptance", base_model=cfg.model_name, batch_size=2,
            epochs=1, warmup_steps=0, seed=19, precision=cfg.precision, compile_model=False,
            checkpoint_target=8, measure_base_kl=True, datasets=[str(uuid4())])
        ids = []
        for interrupted in (False, True):
            response = await client.post(f"/sft?request_id={uuid4()}", headers=headers(["administration"]),
                json=sft.model_dump(mode="json", exclude_unset=True))
            assert response.status_code == 202, response.text
            job_id = UUID(response.json()["run_id"])
            ids.append(job_id)
            if not interrupted:
                expected_config = await execute_existing(runner, job_id)
                continue
            original = training_handler.TrainingPublications.__call__

            def cancel_after_adoption(owner, notice, artifacts):
                """Request actual cancellation only after a real SFT checkpoint is durably adopted and acknowledged."""
                original(owner, notice, artifacts)
                if notice.artifact.format == "model_package_v1" and not notice.final:
                    owner.context.cancel.set()

            with store.session() as db:
                row = db.get(Run, job_id)
                row.status = RunStatus.running
                config, initiating_actor = dict(row.config), str(row.owner_id)
            with monkeypatch.context() as patch:
                patch.setattr(training_handler.TrainingPublications, "__call__", cancel_after_adoption)
                await runner._execute(job_id, RunKind.train, config, initiating_actor)
            with store.session() as db:
                row = db.get(Run, job_id)
                assert row.status == RunStatus.cancelled, row.error
                assert q.checkpoints_of(db, row.training.target_model_id)
        response = await client.post(f"/runs/{ids[1]}/recover?request_id={uuid4()}", headers=headers(["administration"]))
        assert response.status_code == 202, response.text
        recovered_config = await execute_existing(runner, UUID(response.json()["run_id"]))
        expected = verify(storage.model_dir(expected_config["model_name"]))
        recovered = verify(storage.model_dir(recovered_config["model_name"]))
        assert expected.manifest.definition == recovered.manifest.definition == base.manifest.definition
        assert expected.integrity.files["model.pt"].sha256 == recovered.integrity.files["model.pt"].sha256
        assert (base.directory / "tokenizer.json").read_bytes() == base_tokenizer
        assert not list(manager.root.glob("*/ownership.json"))
        frames = registry.get(str(ids[0])).history
        assert any(event.kind == "train" and (event.data.get("diag") or {}).get("base_kl") is not None for event in frames)
