"""Real parent serving/grant adapters with an explicit synthetic worker seam, not process-containment evidence."""

import shutil
import base64
import struct
import threading
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
import torch

import store
from api import serving
from api.training_admission import Capacity
from llm import storage
from llm.architecture.package_integrity import encode_json
from llm.architecture.wire import AttentionMatrix, AttentionSelection, InspectionCatalog
from llm.config import ChatConfig, ChatTurn, EvalConfig
from llm.corpus import bpe
from llm.model.device import Busy
from llm.model.device import DeviceGate
from llm.model.inference_records import AttentionInput, InspectionInput, InspectionRefusal, MEASUREMENT_REQUEST, GenerationInput, GenerationResult, MeasurementInput, MeasurementResult
from llm.runtime.artifacts import ArtifactStore
from llm.runtime.protocol import Completed, Generate, Load, Measure, Tokens
from llm.runtime.supervisor import RequestCancelled, WorkerStopped, WorkerManager
from store import queries as q
from tests.test_architecture_catalog_routes import client
from tests.test_architecture_visibility import headers
from tests.test_modular_rl_worker import pretrained_policy


@pytest.fixture(scope="module")
def actual_package(tmp_path_factory):
    """Train one real tiny package for immutable parent verification; synthetic requests below do no child model execution."""
    return pretrained_policy(tmp_path_factory.mktemp("serving-source"), "standard")[0]


class SyntheticSession:
    """Exercise actual parent lifecycle/backpressure/result validation without pretending to own an OS worker."""

    def __init__(self, root, policy, owner):
        """Create genuine finite private grants and explicit adapter behavior flags."""
        self.worker_id = uuid4()
        self.directory = root / str(self.worker_id)
        self.directory.mkdir()
        self.artifacts = ArtifactStore(self.directory / "artifacts", policy.resources)
        self.policy, self.owner, self.closed = policy, owner, False
        self.loads = 0
        self.generations = 0
        self.finished = threading.Event()

    def run(self, command, *, on_event=None, cancel=None):
        """Return labeled synthetic token/measurement data through the production callback ordering, optionally evicting after result consumption."""
        try:
            if isinstance(command, Load):
                self.loads += 1
                return Completed(type="completed", outcome="ok")
            if isinstance(command, Generate):
                self.finished.clear()
                self.generations += 1
                prepared = GenerationInput.model_validate_json(self.artifacts.read_bytes(command.prompt))
                ids = self.owner.token_ids or bpe.encode(self.owner.tokenizer, "ab")
                if self.owner.invalid_token:
                    ids[-1] = self.owner.tokenizer.get_vocab_size()
                for index, token_id in enumerate(ids):
                    on_event(Tokens(type="tokens", token_ids=(token_id,)))
                    if index == 0 and self.owner.pause:
                        if cancel.wait(5):
                            raise RequestCancelled("synthetic consumer cancelled")
                        pytest.fail("parent did not cancel its synthetic request")
                text = "wrong" if self.owner.invalid_result else bpe.decode(self.owner.tokenizer, ids)
                result = GenerationResult(model_name=prepared.config.model_name, text=text,
                    prompt_tokens=len(prepared.prompt_ids), generated_tokens=len(ids), drained_blocks=0,
                    tool_calls=(), stopped="length", elapsed_secs=0)
                grant = self.artifacts.write_bytes(encode_json(result), "generation_result_v1")
            elif isinstance(command, Measure):
                prepared = MEASUREMENT_REQUEST.validate_json(self.artifacts.read_bytes(command.measurement))
                result = self.observation(prepared, command.pins)
                grant = self.artifacts.write_bytes(encode_json(result), "measurement_result_v1")
            else:
                pytest.fail("unexpected synthetic request")
            completed = Completed(type="completed", outcome="ok", result=grant)
            on_event(completed)
            if self.owner.evict_after_result:
                self.close()
            return completed
        finally:
            self.finished.set()

    def observation(self, prepared, pins):
        """Provide labeled synthetic data for parent boundary tests, never claiming capture or numerical attention evidence."""
        if isinstance(prepared, MeasurementInput):
            return MeasurementResult(model={"id": prepared.config.model_name, "name": prepared.config.model_name,
                "context": 128, "vocab": 320, "parameters": 123}, likelihood={"total": -1.5, "count": 3, "unscored_bytes": 1}, samples=())
        identity = dict(model_name=prepared.model_name, definition_id=pins.definition_id,
            profile_digest=pins.profile_digest, runtime_id=self.worker_id)
        if isinstance(prepared, InspectionInput):
            return InspectionCatalog(**identity, context_window=128, max_tokens=128,
                points=({"id": "a" * 64, "title": "Synthetic test view", "instance_path": "test", "call_path": (0,), "heads": 2},))
        selection = prepared.selection
        labels = tuple(bpe.token_text(self.owner.tokenizer, value) for value in bpe.encode(self.owner.tokenizer, selection.text))
        size = len(labels)
        weights = struct.pack(f"<{size * size}f", *(float(row == column) for row in range(size) for column in range(size)))
        result = AttentionMatrix(**identity, layer=selection.layer if selection.point_id is None else None,
            layers=2 if selection.point_id is None else None, head=selection.head, heads=2, max_tokens=128,
            tokens=labels, shape=(size, size), weights=base64.b64encode(weights).decode(), point_id=selection.point_id or "a" * 64,
            validated_sequence_max=size, evidence_hash="b" * 64)
        invalid = self.owner.invalid_observation
        if invalid == "refused":
            return InspectionRefusal(code="validation")
        if invalid == "runtime":
            return result.model_copy(update={"runtime_id": uuid4()})
        if invalid == "point":
            return result.model_copy(update={"point_id": "c" * 64})
        if invalid == "tokens":
            return result.model_copy(update={"tokens": tuple("wrong" for _ in labels)})
        return result

    def close(self):
        """Retire only this synthetic adapter's files; this is not a guardian cleanup certificate."""
        self.closed = True
        shutil.rmtree(self.directory, ignore_errors=True)


class SyntheticWorkers:
    """Observe parent resident-session selection while production numerical execution is deliberately disabled."""

    def __init__(self, root, tokenizer):
        """Keep one explicit synthetic session list and no independent device gate or process."""
        self.root, self.tokenizer = root, tokenizer
        self.pause = self.invalid_token = self.invalid_result = self.evict_after_result = False
        self.token_ids = None
        self.invalid_observation = None
        self.sessions = []
        self.quarantine = []
        self.gate = SimpleNamespace(quarantine=self.quarantine.append)

    def resident_session(self, policy, backend):
        """Require the actual approved backend name and return one finite synthetic session."""
        assert backend == serving.SERVING_BACKEND
        session = SyntheticSession(self.root, policy, self)
        self.sessions.append(session)
        return session


@pytest.fixture
def parent(actual_package, data_root, store_session, tmp_path, monkeypatch):
    """Register/copy genuine immutable package bytes and connect only the parent to the explicit synthetic worker adapter."""
    model_id = str(uuid4())
    directory = storage.model_dir(model_id)
    shutil.copytree(actual_package.directory, directory)
    with store.session() as db:
        q.upsert_model(db, model_id=UUID(model_id), name="published fixture", published=True)
    root = tmp_path / "workers"
    root.mkdir()
    workers = SyntheticWorkers(root, bpe.load(directory))
    active = serving.ServingModels()
    active.attach_workers(workers)
    monkeypatch.setattr(serving, "capacity", lambda root: Capacity(host_bytes=8 << 30, disk_bytes=8 << 30))

    def forbidden(*args, **kwargs):
        """Detect any parent custom import, tensor load or numerical construction during a serving request."""
        pytest.fail("parent attempted numerical/source execution")

    monkeypatch.setattr("torch.load", forbidden)
    monkeypatch.setattr("llm.architecture.source.SourceLoader.load", forbidden)
    monkeypatch.setattr("llm.orchestrate.inference_worker.InferenceBackend.__init__", forbidden)
    yield active, workers, ChatConfig(model_name=model_id, turns=[ChatTurn(role="user", text="one")], max_tokens=8, seed=17)
    active.evict()


def drain(generator):
    """Collect one actual parent iterator's frames and complete summary, always closing its request lifetime."""
    events = []
    try:
        while True:
            events.append(next(generator))
    except StopIteration as done:
        return events, done.value
    finally:
        generator.close()


def test_parent_reuses_one_resident_and_retires_all_request_grants(parent):
    """Repeated chat and measurement share one immutable model worker while per-request files remain bounded."""
    active, workers, cfg = parent
    first, summary = drain(active.stream(cfg, threading.Event()))
    assert summary["text"] == "ab" and "".join(data["token"] for kind, data in first if kind == "token") == "ab"
    _, repeated = drain(active.stream(cfg, threading.Event()))
    assert repeated == summary and len(workers.sessions) == 1
    session = workers.sessions[0]
    assert session.loads == 1 and session.generations == 2 and not list(session.artifacts.directory.iterdir())
    measured = active.measure(EvalConfig(model_name=cfg.model_name), "one two", [], threading.Event())
    assert measured["likelihood"]["count"] == 3
    assert not list(session.artifacts.directory.iterdir())
    active.evict()
    assert session.closed and not session.directory.exists()


def test_parent_flushes_utf8_without_adding_generated_token_usage(parent):
    """Text-only final decoder output completes the stream while usage still counts only actual emitted vocabulary IDs."""
    active, workers, cfg = parent
    workers.token_ids = [workers.tokenizer.token_to_id("Ã")]
    frames, summary = drain(active.stream(cfg, threading.Event()))
    assert "".join(data["token"] for kind, data in frames if kind == "token") == summary["text"] == "�"
    assert summary["generated_tokens"] == 1


@pytest.mark.parametrize("operation", ["chat", "measure"])
def test_result_is_consumed_before_queue_priority_evicts_its_session(parent, operation):
    """A queued job can remove private grants immediately after completion without losing the already parsed serving result."""
    active, workers, cfg = parent
    workers.evict_after_result = True
    if operation == "chat":
        assert drain(active.stream(cfg, threading.Event()))[1]["text"] == "ab"
    else:
        assert active.measure(EvalConfig(model_name=cfg.model_name), "one", [], threading.Event())["likelihood"]["count"] == 3
    assert workers.sessions[0].closed
    workers.evict_after_result = False
    assert drain(active.stream(cfg, threading.Event()))[1]["text"] == "ab"
    assert len(workers.sessions) == 2


def test_disconnected_consumer_cancels_owned_drain_before_releasing_request(parent):
    """Closing after one frame cancels the pending worker operation and joins its parent thread before another request starts."""
    active, workers, cfg = parent
    workers.pause = True
    iterator = active.stream(cfg, threading.Event())
    assert next(iterator)[0] == "token"
    with pytest.raises(Busy):
        next(active.stream(cfg, threading.Event()))
    iterator.close()
    assert workers.sessions[0].closed and workers.sessions[0].finished.is_set()
    assert active.session is None and not workers.quarantine
    workers.pause = False
    assert drain(active.stream(cfg, threading.Event()))[1]["text"] == "ab"


@pytest.mark.parametrize("invalid", ["invalid_token", "invalid_result"])
def test_parent_rejects_worker_output_outside_actual_vocabulary_or_stream(parent, invalid):
    """Malformed numerical output cannot become successful chat and causes resident retirement."""
    active, workers, cfg = parent
    setattr(workers, invalid, True)
    with pytest.raises(WorkerStopped):
        drain(active.stream(cfg, threading.Event()))
    assert active.session is None and workers.sessions[0].closed


def test_invalid_options_and_insufficient_copy_admission_refuse_before_worker_creation(parent, monkeypatch):
    """Unsupported context, nonfinite sampling and insufficient disk cannot launch custom model execution."""
    active, workers, cfg = parent
    for change in ({"context_multiplier": 2.0}, {"max_tokens": 0}, {"temperature": float("nan")}):
        with pytest.raises(ValueError):
            drain(active.stream(cfg.model_copy(update=change), threading.Event()))
    assert not workers.sessions
    monkeypatch.setattr(serving, "capacity", lambda root: Capacity(host_bytes=8 << 30, disk_bytes=32))
    with pytest.raises(ValueError, match="disk"):
        drain(active.stream(cfg, threading.Event()))
    assert not workers.sessions


def test_disconnect_during_parent_package_preparation_retires_empty_worker(parent, monkeypatch):
    """A cancellation arriving between resident startup and parent archive copying stops preparation before any model load."""
    active, workers, cfg = parent
    cancel = threading.Event()
    original = serving.package_transfer.write_archive

    def disconnect(*args, **kwargs):
        """Signal a real request boundary before delegating to the cancellable archive owner."""
        cancel.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(serving.package_transfer, "write_archive", disconnect)
    with pytest.raises(RequestCancelled):
        drain(active.stream(cfg, cancel))
    assert len(workers.sessions) == 1 and workers.sessions[0].closed and workers.sessions[0].loads == 0
    assert active.session is None


async def test_normal_signed_user_can_chat_and_measure_a_published_saved_model(client, parent, monkeypatch):
    """The public Rust-style service scope can use saved-model execution without receiving authoring/source privileges."""
    active, workers, cfg = parent
    monkeypatch.setattr(serving, "SERVING", active)
    response = await client.post("/chat", headers=headers(["public"]), json=cfg.model_dump(mode="json"))
    assert response.status_code == 200 and response.json()["text"] == "ab"
    measured = await client.post(f"/measure/{uuid4()}", headers=headers(["public"]), json={
        "config": {"model_name": cfg.model_name}, "text": "one two", "prompts": []})
    assert measured.status_code == 200 and measured.json()["model"]["name"] == "published fixture"
    points = await client.get(f"/models/{cfg.model_name}/attention-points", headers=headers(["public"]))
    assert points.status_code == 200 and points.json()["points"][0]["evidence_hash"] is None
    attention = await client.post(f"/models/{cfg.model_name}/attention", headers=headers(["public"]),
        json={"text": "one", "point_id": points.json()["points"][0]["id"], "head": 1})
    assert attention.status_code == 200 and attention.json()["head"] == 1
    assert (await client.get(f"/models/{cfg.model_name}/definition", headers=headers(["public"]))).status_code == 403


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware deferred by user"))])
def test_supervised_parent_serves_reuses_measures_and_disconnects_actual_saved_model(actual_package, data_root, store_session, tmp_path, device):
    """Exercise real guardian/process/admission/grants end to end; sandbox setup denials remain failures, never synthetic passes."""
    model_id = uuid4()
    if device == "cuda":
        source = tmp_path / "cuda-source"
        source.mkdir()
        actual_package = pretrained_policy(source, "standard", device=device)[0]
    shutil.copytree(actual_package.directory, storage.model_dir(str(model_id)))
    with store.session() as db:
        q.upsert_model(db, model_id=model_id, name="supervised saved model", published=True)
    gate = DeviceGate()
    active = serving.ServingModels()
    cfg = ChatConfig(model_name=str(model_id), turns=[ChatTurn(role="user", text="one")], max_tokens=8, seed=19)
    with WorkerManager(tmp_path / "actual-runtime", gate) as workers:
        active.attach_workers(workers)
        try:
            _, result = drain(active.stream(cfg, threading.Event()))
            assert result["model_name"] == str(model_id)
            resident = active.session.worker_id
            assert drain(active.stream(cfg, threading.Event()))[1]["text"] == result["text"]
            assert active.session.worker_id == resident
            measured = active.measure(EvalConfig(model_name=str(model_id), max_tokens=4), "one two three", ["one"], threading.Event())
            assert measured["likelihood"]["count"] > 0 and measured["model"]["context"] == 128
            points = active.inspection_points(str(model_id), threading.Event())
            matrix = active.attention(str(model_id), AttentionSelection(text="one", point_id=points.points[0].id), threading.Event())
            assert matrix.runtime_id == resident and matrix.evidence_hash and len(matrix.tokens) > 0
            iterator = active.stream(cfg.model_copy(update={"seed": 27}), threading.Event())
            try:
                next(iterator)
            except StopIteration:
                pass
            finally:
                iterator.close()
            active.evict()
            assert not workers._sessions
            with gate.admit():
                pass
        finally:
            active.evict()


@pytest.mark.parametrize("invalid", ["runtime", "point", "tokens"])
def test_parent_rejects_stale_attention_identity_and_wrong_actual_axes(parent, invalid):
    """A typed but unrelated observation cannot escape the parent even when its synthetic worker produces valid JSON."""
    active, workers, cfg = parent
    catalog = active.inspection_points(cfg.model_name, threading.Event())
    workers.invalid_observation = invalid
    with pytest.raises(WorkerStopped):
        active.attention(cfg.model_name, AttentionSelection(text="one", point_id=catalog.points[0].id), threading.Event())
    assert active.session is None and workers.sessions[0].closed


def test_observation_result_is_validated_before_queue_eviction_retires_grants(parent):
    """Discovery and attention use the same completion-before-eviction boundary as chat and benchmark measurement."""
    active, workers, cfg = parent
    catalog = active.inspection_points(cfg.model_name, threading.Event())
    workers.evict_after_result = True
    result = active.attention(cfg.model_name, AttentionSelection(text="one", point_id=catalog.points[0].id), threading.Event())
    assert result.runtime_id == catalog.runtime_id and workers.sessions[0].closed
    assert result.validated_sequence_max == len(result.tokens)


async def test_public_attention_refusal_remains_actionable_and_retires_the_failed_resident(client, parent, monkeypatch):
    """A classified numerical refusal becomes a safe 400 without leaking source or retaining a potentially invalid worker."""
    active, workers, cfg = parent
    monkeypatch.setattr(serving, "SERVING", active)
    catalog = active.inspection_points(cfg.model_name, threading.Event())
    workers.invalid_observation = "refused"
    response = await client.post(f"/models/{cfg.model_name}/attention", headers=headers(["public"]),
        json={"text": "one", "point_id": catalog.points[0].id})
    assert response.status_code == 400 and "capture or numerical" in response.json()["detail"]
    assert active.session is None and workers.sessions[0].closed
