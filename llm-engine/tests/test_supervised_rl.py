"""Runnable genuine queue/guardian RL recovery; data and gym HTTP responses use explicit finite transport fixtures."""

import asyncio
from uuid import UUID, uuid4

import httpx
import pytest
import torch

import store
from api import builtin_tokenizer, handlers, rl_effects, training_handler
from api.runner import Runner
from api.runs import Registry
from llm import backend, storage
from llm.architecture.package import verify
from llm.architecture.package_integrity import decode_json, encode_json
from llm.config import RlConfig, TrainConfig
from llm.corpus import remote
from llm.runtime.supervisor import WorkerManager
from store import queries as q
from store.schema import Run, RunKind, RunStatus
from tests.test_architecture_visibility import headers
from tests.test_bounded_hop import Body
from tests.test_supervised_pretraining import CorpusPages, accepted_standard
from tests.test_supervised_sft import enqueue_base
from tests.test_validation_queue import client


class RolloutCorpus(CorpusPages):
    """Provide finite actual prose sufficient for pretraining the inherited 128-token rollout window."""

    def documents(self, identity, *, cap=None):
        """Yield byte-level text through the real parent packer instead of forging model/corpus output."""
        yield from ({"content": "one two three four five " * 32} for _ in range(cap or 4))


def gym_transport(monkeypatch, actor):
    """Keep real parent JWT/size/ownership behavior while supplying finite Rust-shaped gym responses, without claiming live Rust service integration."""
    episodes, calls = {}, []

    def respond(request):
        """Create fresh IDs, record actual step requests and grade siblings independently through parent HTTP effects."""
        path = request.url.path
        body = decode_json(request.content) if request.content else None
        calls.append((path, body))
        if path == "/internal/gym/runs":
            rows = [{"id": str(uuid4()), "index": index, "gym_id": "bash", "task_type": "bash_ls_easy",
                "seed": body["seed_base"] + index // body["group_size"], "prompt": "one", "system": "two", "tools": ["bash"]}
                for index in range(body["count"])]
            episodes.update({row["id"]: row for row in rows})
            result = {"id": str(uuid4()), "episodes": rows}
        elif path.endswith("/finish"):
            result = {"score": float(episodes[path.split("/")[-2]]["index"] % 2 == 0), "answer": "graded"}
        elif path.endswith("/step"):
            result = {"observation": "rendered output", "reward": 0.05, "step_num": 1}
        elif path.endswith("/close"):
            result = {"episode_id": path.split("/")[-2], "status": "closed"}
        else:
            result = {"id": path.split("/")[-2], "status": body["status"]}
        return httpx.Response(200, stream=Body((encode_json(result),)))

    def connect(owner):
        """Require the actual queued actor at every Rust hop, independently of worker-supplied effect payloads."""
        assert owner == actor["id"]
        hop = backend.Hop(owner, url="http://rust.test")
        hop._http.close()
        hop._http = httpx.Client(transport=httpx.MockTransport(respond))
        return hop

    monkeypatch.setattr(rl_effects, "Hop", connect)
    return episodes, calls


async def execute(runner, job_id, actor):
    """Run the actual durable job kind under the real supervisor and require registered completion after cleanup."""
    with store.session() as db:
        run = db.get(Run, job_id)
        run.status = RunStatus.running
        cfg, kind, owner = dict(run.config), run.kind, str(run.owner_id)
    actor["id"] = owner
    await runner._execute(job_id, kind, cfg, owner)
    with store.session() as db:
        run = db.get(Run, job_id)
        assert run.status == RunStatus.done, run.error
        assert q.get_model(db, run.ref_id).definition_id == run.training.definition_id
    return cfg


@pytest.mark.parametrize("device", ["cpu", "cuda"])
async def test_real_queued_rl_recovers_completed_iteration_and_original_reference(client, store_session, tmp_path, monkeypatch, device):
    """Train a genuine base, run signed RL, cancel after actual checkpoint adoption and compare fresh supervised recovery byte-for-byte."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("user-deferred CUDA hardware verification")
    if device == "cpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(remote, "Client", RolloutCorpus)
    revision = accepted_standard()
    tokenizer_id, _raw = builtin_tokenizer.baseline_bytes()
    base = TrainConfig(model_name=str(uuid4()), batch_size=2, warmup_steps=0, compile_model=False,
        precision="fp32" if device == "cpu" else "bf16", checkpoint_target=0,
        tokenizer_id=str(tokenizer_id), datasets=[str(uuid4())], architecture={"revision_id": revision.id,
            "parameters": {"dimensions": 8, "heads": 2, "ff_multiplier": 2, "context_window": 128}})
    base_job = enqueue_base(base, uuid4())
    registry = Registry()
    registry.bind_loop(asyncio.get_running_loop())
    actor = {"id": None}
    episodes, _calls = gym_transport(monkeypatch, actor)
    with WorkerManager(tmp_path / "runtime") as manager:
        runner = Runner(registry, {RunKind.train: handlers.train, RunKind.rl: handlers.reinforce})
        runner.attach_workers(manager)
        await execute(runner, base_job, actor)
        request = RlConfig(model_name="RL complete", base_model=base.model_name, task_types=["bash_ls_easy"],
            group_size=2, tasks_per_batch=1, iterations=2, max_rounds=1, micro_batch=1, max_tokens=8, checkpoint_target=2)
        response = await client.post(f"/rl?request_id={uuid4()}", headers=headers(["administration"]),
            json=request.model_dump(mode="json", exclude_unset=True))
        assert response.status_code == 202, response.text
        normal = await execute(runner, UUID(response.json()["run_id"]), actor)
        response = await client.post(f"/rl?request_id={uuid4()}", headers=headers(["administration"]),
            json={**request.model_dump(mode="json", exclude_unset=True), "model_name": "RL interrupted"})
        assert response.status_code == 202, response.text
        interrupted = UUID(response.json()["run_id"])
        before = training_handler.TrainingPublications.__call__

        def cancel_after_checkpoint(owner, notice, artifacts):
            """Cancel only after actual durable model/evidence registration, retaining the acknowledged completed iteration."""
            before(owner, notice, artifacts)
            if notice.artifact.format == "model_package_v1" and not notice.final:
                owner.context.cancel.set()

        with store.session() as db:
            run = db.get(Run, interrupted)
            run.status = RunStatus.running
            cfg, owner = dict(run.config), str(run.owner_id)
        actor["id"] = owner
        with monkeypatch.context() as patch:
            patch.setattr(training_handler.TrainingPublications, "__call__", cancel_after_checkpoint)
            await runner._execute(interrupted, RunKind.rl, cfg, owner)
        with store.session() as db:
            assert db.get(Run, interrupted).status == RunStatus.cancelled
            checkpoint = q.checkpoints_of(db, UUID(cfg["model_name"]))[-1]
            assert checkpoint.checkpoint_step == 1
        old_ids = set(episodes)
        recovered = await client.post(f"/runs/{interrupted}/recover?request_id={uuid4()}", headers=headers(["administration"]))
        assert recovered.status_code == 202, recovered.text
        resumed = await execute(runner, UUID(recovered.json()["run_id"]), actor)
        expected = verify(storage.model_dir(normal["model_name"]))
        actual = verify(storage.model_dir(resumed["model_name"]))
        assert actual.manifest.definition == expected.manifest.definition
        assert actual.integrity.files["model.pt"].sha256 == expected.integrity.files["model.pt"].sha256
        assert resumed["base_model"] == base.model_name and set(episodes) - old_ids
        assert not list(manager.root.glob("*/ownership.json"))
