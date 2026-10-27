"""Typed worker gym effects, real grant lifetimes and issuance journal without claiming network or process acceptance."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, Thread
from uuid import uuid4

import pytest

from llm.architecture.package_integrity import decode_json, encode_json
from llm.gym.effect_client import EffectClient
from llm.gym.effect_records import CreatedRun, CreateRun
from llm.runtime.artifacts import ArtifactStore
from llm.runtime.effects import EffectBroker, EffectJournal
from llm.runtime.protocol import EffectRequest, RuntimeResources
from llm.runtime.worker import WorkerCancelled


class Context:
    """Expose the actual effect/grant interfaces through an in-process broker, never a fabricated supervisor certificate."""

    def __init__(self, root, dispatch):
        """Own bounded real artifact files, a durable issuance journal and one initiating actor at the parent fixture seam."""
        self.artifacts = ArtifactStore(root / "artifacts", RuntimeResources(process_address_bytes=16 << 30,
            tree_resident_bytes=3 << 30, tree_memory_enforcement="preflight_and_monitor", device="cpu",
            device_allocator_bytes=0, artifact_bytes=8 << 20, total_artifact_bytes=32 << 20))
        self.actor = str(uuid4())
        self.journal = EffectJournal(root / "issued.jsonl")
        self.broker = EffectBroker(self.actor, dispatch, self.journal)
        self.cancelled = False
        self.requests = []

    def check_cancelled(self):
        """Raise the same cooperative signal before another grant or remote operation is issued."""
        if self.cancelled:
            raise WorkerCancelled("effect fixture cancelled")

    def effect(self, operation):
        """Run the real one-outstanding broker and preserve operation identity across its durable issuance boundary."""
        request = EffectRequest(type="effect_request", operation_id=uuid4(), effect=operation)
        self.requests.append(request)
        return self.broker.begin(request, self.artifacts).result(timeout=2)


def public_run(count, *, seed_base=0, group_size=2):
    """Build public task fields plus deliberately unconsumed service metadata to test bounded projection."""
    return {"id": str(uuid4()), "status": "running", "episodes": [{
        "id": str(uuid4()), "index": index, "gym_id": "bash", "task_type": "bash_ls_easy", "seed": seed_base + index // group_size,
        "prompt": "find the file", "system": "answer", "tools": ["bash"], "answer": "PRIVATE_GRADE"}
        for index in range(count)]}


def test_worker_gym_effects_roundtrip_typed_results_and_retire_each_grant(tmp_path):
    """The existing gym domain sees exact public tasks, observations and grades while every consumed grant is removed."""
    calls = []

    def dispatch(actor, request, artifacts):
        """Provide explicitly trusted fixture responses through the real parent broker, projecting task data before granting it."""
        operation = request.effect
        calls.append((actor, operation.operation))
        if operation.operation == "gym_catalogue":
            result = {"gyms": [{"id": "math", "task_types": [{"id": "addition"}]}]}
        elif operation.operation == "gym_create":
            body = CreateRun.model_validate_json(artifacts.read_bytes(operation.payload))
            result = CreatedRun.model_validate_json(encode_json(public_run(body.count))).model_dump(mode="json")
            assert "PRIVATE_GRADE" not in str(result)
        elif operation.operation == "gym_step":
            body = decode_json(artifacts.read_bytes(operation.payload))
            assert body == {"tool": "bash", "argument": "echo 1", "reply": "sampled reply"}
            result = {"observation": "backend-rendered observation", "reward": 0.1, "step_num": 1}
        elif operation.operation == "gym_finish":
            result = {"score": 1.0, "answer": "three"}
        else:
            result = {}
        return artifacts.write_bytes(encode_json(result), "effect_data_v1")

    context = Context(tmp_path, dispatch)
    with EffectClient(context) as client:
        assert client.catalogue()[0]["id"] == "math"
        run_id, episodes = client.create_run(gym_id="bash", model_name=str(uuid4()), task_types=["bash_ls_easy"],
            count=2, seed_base=1, group_size=2)
        assert len(episodes) == 2 and episodes[0].tools == ("bash",)
        assert client.step(episodes[0].id, "bash", "echo 1", "sampled reply").observation == "backend-rendered observation"
        assert client.finish(episodes[0].id, "three").score == 1.0
        client.abandon(episodes[1].id)
        client.set_status(run_id, "done")
    assert all(actor == context.actor for actor, _ in calls)
    assert context.artifacts.total_bytes() == 0
    assert len(context.journal.path.read_text().splitlines()) == 5
    with pytest.raises(WorkerCancelled):
        client.catalogue()


def test_uncertain_write_latches_the_iteration_and_cannot_be_replayed_after_restart(tmp_path):
    """The operation is journaled before the uncertain service call; no subsequent worker call can issue another write."""
    issued = []

    def uncertain(actor, request, artifacts):
        """Represent a service write that may have succeeded before the parent lost its response."""
        issued.append(request.effect.episode_id)
        raise RuntimeError("lost tool response")

    context = Context(tmp_path, uncertain)
    client = EffectClient(context)
    episode = str(uuid4())
    with pytest.raises(RuntimeError, match="completed typed result"):
        client.step(episode, "bash", "command", "reply")
    with pytest.raises(WorkerCancelled):
        client.step(episode, "bash", "command", "reply")
    assert issued == [context.requests[0].effect.episode_id]
    assert context.artifacts.total_bytes() == 0
    restarted = EffectBroker(context.actor, uncertain, EffectJournal(context.journal.path))
    with pytest.raises(ValueError, match="already_issued"):
        restarted.begin(context.requests[0], context.artifacts)
    assert len(issued) == 1


def test_parallel_rollout_calls_keep_one_outstanding_effect_and_bounded_storage(tmp_path):
    """The existing rollout thread pool cannot race IPC receive sequences or accumulate consumed grant files."""
    active, maximum = 0, 0
    lock = Lock()

    def dispatch(actor, operation, artifacts):
        """Observe the actual concurrent broker callback count while returning small finite step results."""
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            return artifacts.write_bytes(encode_json({"reward": 0.0, "step_num": 1}), "effect_data_v1")
        finally:
            with lock:
                active -= 1

    context = Context(tmp_path, dispatch)
    client = EffectClient(context)
    episodes = [str(uuid4()) for _ in range(16)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        replies = list(pool.map(lambda identity: client.step(identity, "bash", "command", "reply"), episodes))
    assert len(replies) == 16 and maximum == 1
    assert len({request.operation_id for request in context.requests}) == 16
    assert context.artifacts.total_bytes() == 0


def test_cancellation_precedes_any_effect_or_grant_creation(tmp_path):
    """A cancelled request cannot create a new gym run or cross the issued-before-write boundary."""
    def forbidden(*args):
        """Detect any service invocation after cooperative cancellation."""
        pytest.fail("cancelled effect was dispatched")

    context = Context(tmp_path, forbidden)
    context.cancelled = True
    with pytest.raises(WorkerCancelled):
        EffectClient(context).create_run(gym_id="math", model_name=str(uuid4()), task_types=["addition"],
            count=2, seed_base=1, group_size=2)
    assert not context.requests and not context.journal.path.exists() and context.artifacts.total_bytes() == 0


@pytest.mark.parametrize("response", [b'{"reward":true}', b'{"reward":1,"reward":2}', b'{"reward":NaN}'])
def test_malformed_effect_results_refuse_without_leaking_grants_or_continuing(tmp_path, response):
    """Strict finite fields and duplicate-safe JSON decoding protect policy returns from malformed service values."""
    def dispatch(actor, operation, artifacts):
        """Return deliberately malformed bytes through a valid bounded effect-data grant."""
        return artifacts.write_bytes(response, "effect_data_v1")

    context = Context(tmp_path, dispatch)
    client = EffectClient(context)
    with pytest.raises(ValueError):
        client.step(str(uuid4()), "bash", "command", "reply")
    with pytest.raises(WorkerCancelled):
        client.catalogue()
    assert context.artifacts.total_bytes() == 0


def test_effect_broker_close_drains_the_real_callback_before_grant_retirement(tmp_path):
    """Stopping a waiter cannot retire resources beneath an issued external callback; closing prevents any new operation."""
    started, release, drained = Event(), Event(), Event()
    failures = []

    def dispatch(actor, operation, artifacts):
        """Keep one actual callback active until the test permits its final response materialization."""
        started.set()
        assert release.wait(2)
        return artifacts.write_bytes(encode_json({}), "effect_data_v1")

    context = Context(tmp_path, dispatch)
    from llm.runtime.protocol import GymAbandonEffect
    request = EffectRequest(type="effect_request", operation_id=uuid4(),
        effect=GymAbandonEffect(operation="gym_abandon", episode_id=uuid4()))
    future = context.broker.begin(request, context.artifacts)
    assert started.wait(1)

    def close():
        """Observe the real broker drain from another owner thread, preserving any failure for the assertion."""
        try:
            context.broker.close(timeout=2)
        except Exception as error:
            failures.append(error)
        finally:
            drained.set()

    thread = Thread(target=close)
    thread.start()
    assert not drained.wait(0.02)
    release.set()
    assert drained.wait(1)
    thread.join()
    assert not failures and future.result().outcome == "completed"
    with pytest.raises(ValueError, match="broker_closed"):
        context.broker.begin(request.model_copy(update={"operation_id": uuid4()}), context.artifacts)
    context.artifacts.remove(future.result().result)


def test_effect_drain_timeout_does_not_claim_the_callback_was_stopped(tmp_path):
    """An unresolved external write remains pending after timeout; its owner must keep resources and admission quarantined."""
    started, release = Event(), Event()

    def dispatch(actor, operation, artifacts):
        """Hold an issued callback past a deliberately short drain allowance."""
        started.set()
        assert release.wait(2)
        return artifacts.write_bytes(encode_json({}), "effect_data_v1")

    context = Context(tmp_path, dispatch)
    from llm.runtime.protocol import GymAbandonEffect
    request = EffectRequest(type="effect_request", operation_id=uuid4(),
        effect=GymAbandonEffect(operation="gym_abandon", episode_id=uuid4()))
    future = context.broker.begin(request, context.artifacts)
    try:
        assert started.wait(1)
        with pytest.raises(TimeoutError):
            context.broker.close(timeout=0.01)
        assert not future.done() and context.artifacts.directory.exists()
    finally:
        release.set()
        future.result(timeout=2)
    context.broker.close(timeout=1)
    context.artifacts.remove(future.result().result)
