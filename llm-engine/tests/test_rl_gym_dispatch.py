"""Actual parent gym authorization, durable ownership, grant retirement and bounded HTTP transport with an in-process backend seam."""

import hashlib
from uuid import UUID, uuid4

import httpx
import jwt
import pytest

from api import rl_effects
from api.rl_effect_log import Authority, GymEffectLog
from llm import backend
from llm.architecture.package_integrity import decode_json, encode_json
from llm.config import PracticeSchedule, RlConfig, TaskWeight
from llm.gym.effect_client import EffectClient
from llm.runtime.worker import WorkerCancelled
from tests.test_bounded_hop import Body
from tests.test_rl_effect_client import Context, public_run


@pytest.fixture
def dispatched(tmp_path, monkeypatch):
    """Connect the actual client, broker, dispatcher and durable logs to finite streamed Rust-shaped responses."""
    cfg = RlConfig(model_name=str(uuid4()), base_model=str(uuid4()), task_types=["bash_ls_easy"],
        tasks_per_batch=1, group_size=2, iterations=2, max_rounds=1)
    context = Context(tmp_path, None)
    authority = Authority(job_id=uuid4(), actor_id=context.actor,
        configuration_sha256=hashlib.sha256(encode_json(cfg.model_dump(mode="json"))).hexdigest())
    log = GymEffectLog(tmp_path / "ownership.jsonl", authority, max_bytes=8 << 20)
    dispatcher = rl_effects.GymDispatcher(cfg, log, response_bytes=8 << 20)
    context.broker.dispatch = dispatcher
    replies = []
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        """Assert write intent is already durable at the transport seam, then deliver a finite body or lose the response."""
        calls.append(request)
        if request.method == "POST":
            active = context.broker.dispatch.log
            assert active.pending is not None or len(active.retirements) > len(active.retired)
            identities = [active.pending.request.operation_id] if active.pending is not None else []
            identities.extend(active.retirements)
            assert all(str(identity) in active.path.read_text() for identity in identities)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        raw = reply if isinstance(reply, bytes) else encode_json(reply)
        return httpx.Response(200, stream=Body((raw,)))

    def connect(owner: str) -> backend.Hop:
        """Keep actual Hop authentication and bounded streaming while replacing only its socket transport."""
        hop = backend.Hop(owner, url="http://rust.test", service_secret="test-secret")
        hop._http.close()
        hop._http = httpx.Client(transport=httpx.MockTransport(respond))
        return hop

    monkeypatch.setattr(rl_effects, "Hop", connect)
    yield context, cfg, log, replies, calls
    context.broker.close(timeout=2)


def create(client: EffectClient, cfg: RlConfig, *, seed_base: int | None = None):
    """Request the exact configured finite training batch through the worker-facing gym interface."""
    return client.create_run(gym_id=cfg.gym_id, model_name=cfg.base_model,
        task_types=[cfg.task_types[0].id], count=cfg.tasks_per_batch * cfg.group_size,
        seed_base=cfg.seed_base if seed_base is None else seed_base, group_size=cfg.group_size)


def restore(log: GymEffectLog) -> GymEffectLog:
    """Reopen actual fsynced metadata using the same queued authority and explicit byte allowance."""
    return GymEffectLog(log.path, log.authority, max_bytes=log.max_bytes)


def test_parent_dispatch_projects_public_fields_and_owns_each_resource(dispatched):
    """Only this job's returned IDs/tools can reach Rust, and durable ownership precedes the worker's acknowledgement."""
    context, cfg, log, replies, calls = dispatched
    raw = public_run(2)
    replies.append(raw)
    with EffectClient(context) as client:
        run_id, episodes = create(client, cfg)
        durable = restore(log)
        assert UUID(run_id) in durable.runs and durable.creations["train"] == 1
        assert all(UUID(episode.id) in durable.episodes for episode in episodes)
        assert durable.pending is None
        assert "PRIVATE_GRADE" not in str(episodes)
        replies.append({"observation": "observed output", "reward": 0.1, "step_num": 1})
        assert client.step(episodes[0].id, "bash", "echo UNIQUE_COMMAND", "UNIQUE_REPLY").reward == 0.1
        replies.append({"score": 1.0, "answer": "private answer", "ground_truth": "UNEXPOSED"})
        assert client.finish(episodes[0].id, "final reply").score == 1.0
        replies.append({"episode_id": episodes[1].id, "status": "closed"})
        client.abandon(episodes[1].id)
        replies.append({"id": run_id, "status": "done"})
        client.set_status(run_id, "done")
    durable = restore(log)
    assert durable.runs[UUID(run_id)] and all(episode.terminal for episode in durable.episodes.values())
    assert durable.episodes[UUID(episodes[0].id)].steps == 1
    assert context.artifacts.total_bytes() == 0
    persisted = log.path.read_text()
    assert all(text not in persisted for text in ("UNIQUE_COMMAND", "UNIQUE_REPLY", "observed output", "private answer", "find the file"))
    for request in calls:
        claims = jwt.decode(request.headers["Authorization"][7:], "test-secret", algorithms=["HS256"], audience="backend")
        assert claims["act"] == context.actor
    assert len(calls) == 5


@pytest.mark.parametrize("operation", ["step", "finish", "abandon", "status"])
def test_foreign_resource_ids_never_reach_rust(dispatched, operation):
    """Worker-selected UUIDs cannot authorize service writes, even though they pass the runtime wire schema."""
    context, _, log, _, calls = dispatched
    client = EffectClient(context)
    with pytest.raises(RuntimeError, match="completed typed result"):
        if operation == "step":
            client.step(str(uuid4()), "bash", "echo forbidden", "reply")
        elif operation == "finish":
            client.finish(str(uuid4()), "reply")
        elif operation == "abandon":
            client.abandon(str(uuid4()))
        else:
            client.set_status(str(uuid4()), "done")
    assert calls == [] and log.pending is None
    with pytest.raises(WorkerCancelled):
        client.catalogue()


@pytest.mark.parametrize("violation", ["tool", "rounds", "finished", "run_done"])
def test_parent_enforces_tools_rounds_and_terminal_fences(dispatched, violation):
    """A new operation UUID does not bypass ownership, the round budget or terminal episode/run transitions."""
    context, cfg, log, replies, calls = dispatched
    replies.append(public_run(2))
    client = EffectClient(context)
    run_id, episodes = create(client, cfg)
    episode = episodes[0].id
    if violation == "rounds":
        replies.append({"observation": "", "reward": 0.0, "step_num": 1})
        client.step(episode, "bash", "echo once", "reply")
    elif violation == "finished":
        replies.append({"score": 0.0, "answer": ""})
        client.finish(episode, "reply")
    elif violation == "run_done":
        replies.append({"id": run_id, "status": "done"})
        client.set_status(run_id, "done")
    issued = len(calls)
    with pytest.raises(RuntimeError):
        client.step(episode, "unauthorized" if violation == "tool" else "bash", "echo forbidden", "reply")
    assert len(calls) == issued and log.pending is None


@pytest.mark.parametrize("lost", ["create", "step", "finish", "abandon", "status"])
def test_lost_responses_fence_the_job_across_parent_restart(dispatched, lost):
    """Every uncertain write remains issued; reopening the durable job refuses a new operation rather than repeating a command."""
    context, cfg, log, replies, calls = dispatched
    client = EffectClient(context)
    if lost != "create":
        replies.append(public_run(2))
        run_id, episodes = create(client, cfg)
    replies.append(httpx.ReadError("lost after Rust may have applied the write"))
    with pytest.raises(RuntimeError):
        if lost == "create":
            create(client, cfg)
        elif lost == "step":
            client.step(episodes[0].id, "bash", "echo once", "reply")
        elif lost == "finish":
            client.finish(episodes[0].id, "reply")
        elif lost == "abandon":
            client.abandon(episodes[0].id)
        else:
            client.set_status(run_id, "done")
    durable = restore(log)
    assert durable.pending is not None
    if lost in {"finish", "abandon"}:
        assert durable.episodes[UUID(episodes[0].id)].terminal
    if lost == "status":
        assert durable.runs[UUID(run_id)]
    context.broker.dispatch = rl_effects.GymDispatcher(cfg, durable, response_bytes=8 << 20)
    issued = len(calls)
    with pytest.raises(RuntimeError):
        create(EffectClient(context), cfg)
    assert len(calls) == issued and context.artifacts.total_bytes() == 0


def test_recovery_job_cannot_use_previous_jobs_episode_ids(dispatched, tmp_path):
    """A fresh recovery authority starts without resource grants from the interrupted job."""
    context, cfg, log, replies, calls = dispatched
    replies.append(public_run(2))
    _, episodes = create(EffectClient(context), cfg)
    authority = log.authority.model_copy(update={"job_id": uuid4()})
    fresh = GymEffectLog(tmp_path / "recovery.jsonl", authority, max_bytes=8 << 20)
    context.broker.dispatch = rl_effects.GymDispatcher(cfg, fresh, response_bytes=8 << 20)
    with pytest.raises(RuntimeError):
        EffectClient(context).step(episodes[0].id, "bash", "echo replay", "reply")
    assert len(calls) == 1 and fresh.pending is None


@pytest.mark.parametrize("change", ["actor", "job", "configuration", "partial", "duplicate", "budget"])
def test_durable_ownership_refuses_foreign_or_corrupt_evidence(dispatched, change):
    """Restart cannot silently adopt another job's ledger, incomplete evidence, duplicate fields or unadmitted disk usage."""
    _, _, log, _, _ = dispatched
    authority = log.authority
    budget = log.max_bytes
    if change == "actor":
        authority = authority.model_copy(update={"actor_id": uuid4()})
    elif change == "job":
        authority = authority.model_copy(update={"job_id": uuid4()})
    elif change == "configuration":
        authority = authority.model_copy(update={"configuration_sha256": "1" * 64})
    elif change == "partial":
        with log.path.open("ab") as output:
            output.write(b'{"type":')
    elif change == "duplicate":
        log.path.write_bytes(log.path.read_bytes().replace(b'{"actor_id":', b'{"actor_id":"ignored","actor_id":'))
    else:
        budget = 1
    with pytest.raises(ValueError):
        GymEffectLog(log.path, authority, max_bytes=budget)


def test_wrong_broker_actor_is_refused_before_any_http_request(dispatched):
    """The dispatcher's queued actor binding is checked independently of the runtime broker's own actor field."""
    context, cfg, _, _, calls = dispatched
    context.broker.actor = str(uuid4())
    with pytest.raises(RuntimeError):
        create(EffectClient(context), cfg)
    assert calls == []


@pytest.mark.parametrize("response", [b'{"id":1,"id":2}', b'{"reward":NaN}', {"id": str(uuid4()), "episodes": []}])
def test_malformed_creation_response_retains_uncertain_intent(dispatched, response):
    """Invalid remote data cannot become a completed creation or resource grant, and the issued write stays fenced."""
    context, cfg, log, replies, calls = dispatched
    replies.append(response)
    with pytest.raises(RuntimeError):
        create(EffectClient(context), cfg)
    assert restore(log).pending is not None and len(calls) == 1
    assert log.runs == {} and context.artifacts.total_bytes() == 0


def test_creation_schedule_budget_survives_restart(dispatched):
    """A worker cannot create more training batches than queued, including by reopening parent dispatch state."""
    context, cfg, log, replies, calls = dispatched
    client = EffectClient(context)
    for iteration in range(cfg.iterations):
        replies.append(public_run(2, seed_base=iteration))
        create(client, cfg, seed_base=cfg.seed_base + iteration * cfg.tasks_per_batch)
    durable = restore(log)
    context.broker.dispatch = rl_effects.GymDispatcher(cfg, durable, response_bytes=8 << 20)
    with pytest.raises(RuntimeError):
        create(EffectClient(context), cfg)
    assert len(calls) == cfg.iterations and durable.creations["train"] == cfg.iterations


def test_parent_ownership_budget_refuses_before_rust_write(dispatched):
    """Exhausted durable evidence admission stops issuance rather than allowing an unrecorded service operation."""
    context, cfg, log, _, calls = dispatched
    log.max_bytes = log.path.stat().st_size
    with pytest.raises(RuntimeError):
        create(EffectClient(context), cfg)
    assert calls == [] and log.pending is None


@pytest.mark.parametrize("change", ["gym_id", "model_name", "task_types", "count", "seed_base", "task_weights"])
def test_worker_cannot_expand_queued_creation_authority(dispatched, change):
    """Typed but unauthorized model/lane/task/batch/seed/weight values are rejected before external issuance."""
    context, cfg, log, _, calls = dispatched
    body = {"gym_id": cfg.gym_id, "model_name": cfg.base_model, "task_types": ["bash_ls_easy"],
        "count": 2, "seed_base": 0, "group_size": 2}
    body[change] = {"gym_id": "math", "model_name": str(uuid4()), "task_types": ["unselected"],
        "count": 4, "seed_base": cfg.holdout_base, "task_weights": {"bash_ls_easy": 2.0}}[change]
    with pytest.raises(RuntimeError):
        EffectClient(context).create_run(**body)
    assert calls == [] and log.pending is None


@pytest.mark.parametrize("operation", ["abandon", "status"])
def test_mismatched_terminal_acknowledgements_remain_uncertain(dispatched, operation):
    """A response naming another resource does not clear the issued terminal transition."""
    context, cfg, log, replies, calls = dispatched
    replies.append(public_run(2))
    client = EffectClient(context)
    run_id, episodes = create(client, cfg)
    replies.append({"id": str(uuid4()), "episode_id": str(uuid4()), "status": "closed" if operation == "abandon" else "done"})
    with pytest.raises(RuntimeError):
        if operation == "abandon":
            client.abandon(episodes[0].id)
        else:
            client.set_status(run_id, "done")
    assert restore(log).pending is not None and len(calls) == 2


def test_created_ids_are_durable_even_if_the_worker_grant_cannot_be_delivered(dispatched, monkeypatch):
    """Creation response ownership survives acknowledgement loss; retaining known IDs supports later parent reconciliation."""
    context, cfg, log, replies, calls = dispatched
    raw = public_run(2)
    replies.append(raw)
    write = context.artifacts.write_bytes

    def fail_creation_grant(data, format):
        """Lose only the parent result grant after actual Rust response validation and durable ID acquisition."""
        if "episodes" in decode_json(data):
            raise OSError("grant delivery lost")
        return write(data, format)

    monkeypatch.setattr(context.artifacts, "write_bytes", fail_creation_grant)
    with pytest.raises(RuntimeError):
        create(EffectClient(context), cfg)
    durable = restore(log)
    assert UUID(raw["id"]) in durable.runs
    assert set(durable.episodes) == {UUID(episode["id"]) for episode in raw["episodes"]}
    assert len(calls) == 1 and context.artifacts.total_bytes() == 0


def test_general_sampling_weights_and_source_tasks_are_preserved(dispatched):
    """General creation forwards the configured full pool and weights instead of pre-drawing a biased flat pool."""
    context, cfg, log, replies, calls = dispatched
    cfg = cfg.model_copy(update={"gym_id": "general", "gym_weights": {"bash": 2.0}})
    authority = log.authority.model_copy(update={"job_id": uuid4(),
        "configuration_sha256": hashlib.sha256(encode_json(cfg.model_dump(mode="json"))).hexdigest()})
    general = GymEffectLog(log.path.with_name("general.jsonl"), authority, max_bytes=8 << 20)
    context.broker.dispatch = rl_effects.GymDispatcher(cfg, general, response_bytes=8 << 20)
    replies.append(public_run(2))
    client = EffectClient(context)
    client.create_run(gym_id="general", model_name=cfg.base_model, task_types=["bash_ls_easy"], count=2,
        seed_base=0, group_size=2, gym_weights={"bash": 2.0}, task_weights={"bash_ls_easy": 1.0})
    sent = decode_json(calls[0].content)
    assert sent["task_weights"] == {"bash_ls_easy": 1.0} and sent["gym_weights"] == {"bash": 2.0}
    assert sent["kind"] == "rl" and sent["task_types"] == ["bash_ls_easy"]


def test_curriculum_allows_both_initial_and_final_fixed_tests(dispatched):
    """The queued exam allowance preserves the existing before/after fixed-test pair and refuses extra test runs."""
    context, cfg, log, replies, calls = dispatched
    cfg = cfg.model_copy(update={"gym_id": "math", "task_types": [TaskWeight(id="math_addition_easy")],
        "curriculum": PracticeSchedule(test_samples=1)})
    authority = log.authority.model_copy(update={"job_id": uuid4(),
        "configuration_sha256": hashlib.sha256(encode_json(cfg.model_dump(mode="json"))).hexdigest()})
    exams = GymEffectLog(log.path.with_name("exams.jsonl"), authority, max_bytes=8 << 20)
    context.broker.dispatch = rl_effects.GymDispatcher(cfg, exams, response_bytes=8 << 20)
    client = EffectClient(context)
    for _ in range(2):
        raw = public_run(1)
        raw["episodes"][0].update(gym_id="math", task_type="math_addition_easy", tools=[])
        replies.append(raw)
        client.create_run(gym_id="math", model_name=cfg.base_model, task_types=["math_addition_easy"],
            count=1, seed_base=cfg.holdout_base, group_size=1, split="test")
    assert restore(exams).creations["test"] == 2
    with pytest.raises(RuntimeError):
        client.create_run(gym_id="math", model_name=cfg.base_model, task_types=["math_addition_easy"],
            count=1, seed_base=cfg.holdout_base, group_size=1, split="test")
    assert len(calls) == 2


def test_parent_cleanup_closes_known_resources_without_replaying_an_uncertain_step(dispatched):
    """After callback draining, new close/status intents retire known resources while the lost tool write keeps its original fence."""
    context, cfg, log, replies, calls = dispatched
    replies.append(public_run(2))
    client = EffectClient(context)
    run_id, episodes = create(client, cfg)
    replies.append(httpx.ReadError("lost tool response"))
    with pytest.raises(RuntimeError):
        client.step(episodes[0].id, "bash", "echo once", "reply")
    original = log.pending
    replies.extend([{"episode_id": episode.id, "status": "closed"} for episode in episodes])
    replies.append({"id": run_id, "status": "abandoned"})
    context.broker.close(timeout=2)
    context.broker.dispatch.retire_resources()
    durable = restore(log)
    assert durable.pending == original and len(durable.retired) == len(durable.retirements) == 3
    assert all(episode.terminal for episode in durable.episodes.values()) and durable.runs[UUID(run_id)]
    assert len([call for call in calls if call.url.path.endswith("/step")]) == 1
    assert len([call for call in calls if call.url.path.endswith("/close")]) == 2
    issued = len(calls)
    rl_effects.GymDispatcher(cfg, durable, response_bytes=8 << 20).retire_resources()
    assert len(calls) == issued


def test_uncertain_parent_cleanup_is_itself_never_retried_after_restart(dispatched):
    """Lost close acknowledgements fence that resource durably while independent known resources can still be retired."""
    context, cfg, log, replies, calls = dispatched
    replies.append(public_run(2))
    run_id, episodes = create(EffectClient(context), cfg)
    replies.extend([httpx.ReadError("close response lost"), {"episode_id": episodes[1].id, "status": "closed"},
        {"id": run_id, "status": "abandoned"}])
    context.broker.close(timeout=2)
    context.broker.dispatch.retire_resources()
    durable = restore(log)
    assert len(durable.retirements) == 3 and len(durable.retired) == 2 and durable.pending is None
    assert len(calls) == 4
    context.broker.dispatch = rl_effects.GymDispatcher(cfg, durable, response_bytes=8 << 20)
    context.broker.dispatch.retire_resources()
    assert len(calls) == 4
    with pytest.raises(ValueError):
        durable.require_settled()


def test_parent_cleanup_does_not_repeat_an_uncertain_finish(dispatched):
    """A finish whose response was lost keeps its terminal fence; cleanup only closes its untouched sibling and abandons the run."""
    context, cfg, log, replies, calls = dispatched
    replies.append(public_run(2))
    client = EffectClient(context)
    run_id, episodes = create(client, cfg)
    replies.append(httpx.ReadError("finish response lost"))
    with pytest.raises(RuntimeError):
        client.finish(episodes[0].id, "reply")
    replies.extend([{"episode_id": episodes[1].id, "status": "closed"}, {"id": run_id, "status": "abandoned"}])
    context.broker.close(timeout=2)
    context.broker.dispatch.retire_resources()
    assert len(calls) == 4
    assert len([call for call in calls if call.url.path.endswith("/finish")]) == 1
    assert [call.url.path for call in calls if call.url.path.endswith("/close")] == [f"/internal/gym/episodes/{episodes[1].id}/close"]
    assert restore(log).pending is not None


def test_parent_cleanup_cannot_repeat_creation_when_its_resource_id_was_lost(dispatched):
    """An unknown created run cannot be rediscovered by repeating creation; its durable uncertainty remains for reconciliation."""
    context, cfg, log, replies, calls = dispatched
    replies.append(httpx.ReadError("create response lost"))
    with pytest.raises(RuntimeError):
        create(EffectClient(context), cfg)
    context.broker.close(timeout=2)
    context.broker.dispatch.retire_resources()
    assert len(calls) == 1 and restore(log).pending is not None
    assert log.runs == {}


def test_parent_restart_cleanup_does_not_require_surviving_job_configuration(dispatched):
    """The durable actor/resource header can retire known sandboxes after job-history pruning without importing source or replaying writes."""
    context, cfg, log, replies, calls = dispatched
    replies.append(public_run(2))
    run_id, episodes = create(EffectClient(context), cfg)
    context.broker.close(timeout=2)
    durable = GymEffectLog.open_existing(log.path, max_bytes=8 << 20)
    owner = rl_effects.GymResourceOwner(durable, response_bytes=8 << 20)
    context.broker.dispatch = owner
    replies.extend([{"episode_id": episode.id, "status": "closed"} for episode in episodes])
    replies.append({"id": run_id, "status": "abandoned"})
    owner.retire_resources()
    assert len(calls) == 4 and len(durable.retired) == 3
    assert durable.authority.actor_id == UUID(context.actor)


def test_recovery_authority_limits_seed_position_and_remaining_iterations(dispatched):
    """A completed source boundary authorizes only its next group seed and the actual remaining iteration count."""
    context, cfg, log, replies, calls = dispatched
    authority = log.authority.model_copy(update={"job_id": uuid4(), "first_iteration": 1, "training_group_start": 1})
    recovered = GymEffectLog(log.path.with_name("recovered.jsonl"), authority, max_bytes=8 << 20)
    context.broker.dispatch = rl_effects.GymDispatcher(cfg, recovered, response_bytes=8 << 20)
    replies.append(public_run(2, seed_base=1))
    client = EffectClient(context)
    create(client, cfg, seed_base=1)
    with pytest.raises(RuntimeError):
        create(client, cfg, seed_base=2)
    assert len(calls) == 1 and recovered.creations["train"] == 1


def test_worker_cannot_repeat_the_previous_completed_task_seed(dispatched):
    """A fresh operation UUID cannot authorize a second training draw from the same completed group position."""
    context, cfg, log, replies, calls = dispatched
    replies.append(public_run(2))
    client = EffectClient(context)
    create(client, cfg)
    with pytest.raises(RuntimeError):
        create(client, cfg)
    assert len(calls) == 1 and log.pending is None


@pytest.mark.parametrize("field,value", [("seed", 1), ("prompt", "another question"),
    ("system", "different task instructions"), ("tools", ["another-tool"])])
def test_created_training_siblings_must_share_one_requested_question(dispatched, field, value):
    """Valid UUIDs cannot turn mismatched questions or seeds into one group-baseline policy update."""
    context, cfg, log, replies, calls = dispatched
    raw = public_run(2)
    raw["episodes"][1][field] = value
    replies.append(raw)
    with pytest.raises(RuntimeError):
        create(EffectClient(context), cfg)
    assert len(calls) == 1 and log.pending is not None and not log.runs
