"""Actual inherited RL arithmetic/packages/recovery through parent effects; these in-process tests do not certify supervision."""

import hashlib
from types import SimpleNamespace
from uuid import uuid4

import httpx
import numpy as np
import pytest
import torch
import yaml

from api import rl_effects
from api.rl_effect_log import Authority, GymEffectLog
from llm import backend
from llm.architecture import package_transfer
from llm.architecture.package_integrity import decode_json, encode_json
from llm.architecture.spec import RevisionRef, RevisionSubject
from llm.architecture.tokenizer_binding import binding_from_bytes
from llm.architecture.validation_records import PinnedSource
from llm.architecture.wire import ArchitectureSelection
from llm.config import PracticeSchedule, RlConfig
from llm.corpus.baseline import byte_alphabet
from llm.corpus import special
from llm.corpus.spans import DocSpan, Spans
from llm.model import checkpoint
from llm.model.device import DeviceGate
from llm.runtime.effects import EffectBroker, EffectJournal
from llm.runtime.protocol import EffectRequest, Train
from llm.runtime.worker import WorkerCancelled
from llm.runtime.supervisor import WorkerManager
from llm.train import data_identity
from llm.train.modular_records import CorpusPlan, TrainingPreparation, TrainingPublication
from llm.train.modular_worker import TrainingBackend
from llm.train.model_preflight import rl_profile, training_profile
from llm.train.plan import DataPlan
from llm.train.rl_records import RlPreparation
from llm.train.rl_worker import RlBackend
from llm.train.sft_tokenizer import marker_history
from llm.train.warm_start import untrained_marker_ids
from tests.test_architecture_core import revision
from tests.test_bounded_hop import Body
from tests.test_modular_sft_worker import allow_trusted_tensor_reader
from tests.test_modular_worker import NumericalContext, preparation
from tests.test_token_rows import ROW_SOURCE, row_recipe


def pretrained_policy(tmp_path, name, *, known_marker_history=False, device="cpu"):
    """Run genuine tiny pretraining with a byte-complete tokenizer and a window large enough for actual RL grammar/sampling."""
    context = NumericalContext(tmp_path / "base")
    if device == "cuda":
        from llm.train import memory

        context.device = device
        resources = context.bootstrap.policy.resources.model_copy(update={"device": device,
            "device_allocator_bytes": memory.budget_bytes(device)})
        context.bootstrap.policy = context.bootstrap.policy.model_copy(update={"resources": resources})
    prep, command = preparation(context, name="standard" if name == "custom" else name, checkpoint_target=0, device=device)
    tokenizer = byte_alphabet()
    raw = tokenizer.to_str().encode()
    binding = binding_from_bytes(uuid4(), raw)
    parameters = {**prep.validation.request.parameters, "context_window": 128}
    cfg = prep.config.model_copy(update={"architecture": prep.config.architecture.model_copy(update={"parameters": parameters})})
    snapshot = prep.validation.model_copy(update={"request": prep.validation.request.model_copy(update={
        "parameters": parameters, "tokenizer": binding, "context_window": 128, "profile": training_profile(cfg, 128, device)})})
    if name == "custom":
        source = ROW_SOURCE.split("    def initialize_token_rows")[0].replace('capabilities=("token_rows",)', 'capabilities=()')
        source = source.replace("self.readout = nn.Linear(width, vocab, bias=False)",
            "self.readout = nn.Linear(width, vocab, bias=False)\n        self.noise = nn.Dropout(0.75)")
        source = source.replace("self.readout(self.lookup(tokens))", "self.readout(self.noise(self.lookup(tokens)))")
        block = revision(source)
        recipe = row_recipe(block.reference).model_copy(update={"context_window": 128})
        text = yaml.safe_dump(recipe.model_dump(mode="json", exclude_none=True))
        reference = RevisionRef(kind="architecture", resource_id=uuid4(), revision_id=uuid4(),
            content_hash=hashlib.sha256(text.encode()).hexdigest())
        snapshot = snapshot.model_copy(update={"recipe": text, "dependencies": (PinnedSource(reference=block.reference, source=source),),
            "builtins": (), "request": snapshot.request.model_copy(update={"parameters": {},
                "subject": RevisionSubject(kind="revision", reference=reference)})})
        cfg = cfg.model_copy(update={"architecture": ArchitectureSelection(revision_id=reference.revision_id, parameters={})})
    corpus = np.arange(513, dtype=np.uint16) % tokenizer.get_vocab_size()
    seen = None
    if known_marker_history:
        memory = [tokenizer.token_to_id(marker) for marker in (special.MEMORY_OPEN, special.MEMORY_CLOSE)]
        corpus[np.isin(corpus, memory)] = tokenizer.encode("x").ids[0]
        markers = {tokenizer.token_to_id(marker) for marker in special.block_names()} - {None}
        seen = tuple(sorted(set(corpus.tolist()) & markers))
    spans = Spans.of([DocSpan("fixture", 0, len(corpus))])
    span_path = context.directory / "rl-base-spans.npy"
    spans.save(span_path)
    plan = DataPlan(documents=spans, eot_id=0, shuffle=True, seed=cfg.seed)
    prep = TrainingPreparation(config=cfg, validation=snapshot, tokenizer=context.artifacts.write_bytes(raw, "tokenizer_v1"),
        data=CorpusPlan(tokens=len(corpus), spans=context.artifacts.stage_file(span_path, "span_table_v1"),
            names=("fixture",), stages=(), eot_id=0, corpus_key="rl-base", fingerprint=data_identity.fingerprint(corpus, plan=plan),
            shuffle=True, seed=cfg.seed, special_ids_seen=seen))
    command = Train(type="train", job_id=context.request_id, training_kind="pretrain",
        corpus=context.artifacts.write_bytes(corpus.tobytes(), "corpus_v1"),
        config=context.artifacts.write_bytes(prep.model_dump_json().encode(), "training_config_v1"))
    active = TrainingBackend(context)
    try:
        active.handle(command, context)
    finally:
        active.finish_request()
    return context.packages[-1], prep


class RlContext(NumericalContext):
    """Connect actual numerical publication to the real parent effect broker without fabricating process cleanup evidence."""

    def __init__(self, root):
        """Create finite artifacts and separate effect/publication evidence for this trusted arithmetic fixture."""
        super().__init__(root)
        self.actor = str(uuid4())
        self.requests = []
        self.broker = None
        self.cancel_at_checkpoint = False
        self.stop_after_finishes: int | None = None

    def effect(self, operation):
        """Serialize one genuine broker request and await its finite parent callback result."""
        self.check_cancelled()
        request = EffectRequest(type="effect_request", operation_id=uuid4(), effect=operation)
        self.requests.append(request)
        return self.broker.begin(request, self.artifacts).result(timeout=2)

    def publish(self, artifact, *, completed_step, final=False):
        """Retain complete actual package output, optionally cancelling after the first recovery generation was acknowledged."""
        recovery = artifact.format == "model_package_v1" and not final
        super().publish(artifact, completed_step=completed_step, final=final)
        if recovery and self.cancel_at_checkpoint:
            self.cancelled = True


def rl_preparation(context, saved, original, *, resume=None, reference=None, curriculum=False):
    """Freeze the original saved definition/tokenizer plus optional complete recovery and original-base grants."""
    cfg = RlConfig(model_name=str(uuid4()), base_model=original.config.model_name, resume_from=resume,
        gym_id="math" if curriculum else "bash", task_types=["math_addition_easy" if curriculum else "fixture"],
        curriculum=PracticeSchedule(check_every=1, test_samples=1) if curriculum else None,
        tasks_per_batch=1, group_size=2, iterations=2, max_rounds=1,
        micro_batch=1, max_tokens=8, top_k=0, top_p=1.0, lr=0.001, checkpoint_target=2)
    metadata = TrainingPublication.model_validate_json((saved.directory / "meta.json").read_bytes())
    context.first_iteration = metadata.generation.step if resume is not None else 0
    context.training_group_start = ((metadata.result.get("curriculum") or {}).get("training_groups", 0)
        if curriculum else context.first_iteration * cfg.tasks_per_batch)
    context.request_id = uuid4()
    snapshot = original.validation.model_copy(update={"run_id": context.request_id,
        "request": original.validation.request.model_copy(update={"profile": rl_profile(cfg, 128, "cpu")})})
    raw = (saved.directory / "tokenizer.json").read_bytes()
    frozen = None
    if reference is not None:
        path = context.directory / "reference.tar"
        package_transfer.write_archive(reference.directory, path)
        frozen = context.artifacts.stage_file(path, "model_package_v1")
    prep = RlPreparation(request=cfg, validation=snapshot, tokenizer=context.artifacts.write_bytes(raw, "tokenizer_v1"), reference=frozen)
    archive = context.directory / "policy.tar"
    package_transfer.write_archive(saved.directory, archive)
    command = Train(type="train", job_id=context.request_id, training_kind="rl",
        config=context.artifacts.write_bytes(prep.model_dump_json().encode(), "training_config_v1"),
        recovery=context.artifacts.stage_file(archive, "model_package_v1"))
    return prep, command


def bind_parent(context, cfg, monkeypatch, *, prompt="one"):
    """Use real dispatcher authorization and bounded Hop transport, with only Rust's external service responses supplied locally."""
    authority = Authority(job_id=context.request_id, actor_id=context.actor,
        configuration_sha256=hashlib.sha256(encode_json(cfg.model_dump(mode="json"))).hexdigest(),
        first_iteration=context.first_iteration, training_group_start=context.training_group_start)
    log = GymEffectLog(context.directory / "ownership.jsonl", authority, max_bytes=8 << 20)
    dispatcher = rl_effects.GymDispatcher(cfg, log, response_bytes=8 << 20)
    context.broker = EffectBroker(context.actor, dispatcher, EffectJournal(context.directory / "issued.jsonl"))
    calls, episodes = [], {}

    def respond(request):
        """Create fresh episode IDs, render observations and alternate sibling grades through real effect HTTP requests."""
        calls.append((request.method, request.url.path, decode_json(request.content) if request.content else None))
        body = calls[-1][2]
        path = request.url.path
        if path == "/internal/gyms":
            result = [{"id": cfg.gym_id, "generator_version": 1, "task_types": [
                {"id": entry.id, "label": entry.id, "family": "arithmetic", "level": 1, "prerequisites": []}
                for entry in cfg.task_types]}]
        elif path == "/internal/gym/runs":
            run_id = str(uuid4())
            rows = [{"id": str(uuid4()), "index": index, "gym_id": cfg.gym_id,
                "task_type": body["task_types"][(index // body["group_size"]) % len(body["task_types"])],
                "seed": body["seed_base"] + index // body["group_size"], "prompt": prompt, "system": "two",
                "tools": [] if cfg.gym_id == "math" else ["bash"], "split": body["split"]}
                for index in range(body["count"])]
            episodes.update({row["id"]: row for row in rows})
            result = {"id": run_id, "episodes": rows}
        elif path.endswith("/finish"):
            episode = episodes[path.split("/")[-2]]
            result = {"score": float(episode["split"] != "train" or episode["index"] % 2 == 0), "answer": "graded"}
            finishes = sum(call[1].endswith("/finish") for call in calls)
            if context.stop_after_finishes is not None and finishes >= context.stop_after_finishes:
                context.cancelled = True
        elif path.endswith("/step"):
            result = {"observation": "rendered output", "reward": 0.05, "step_num": 1}
        elif path.endswith("/status"):
            result = {"id": path.split("/")[-2], "status": body["status"]}
        else:
            raise AssertionError(f"unexpected fixture Rust route: {path}")
        return httpx.Response(200, stream=Body((encode_json(result),)))

    def connect(actor):
        """Keep actual JWT minting/response bounds while replacing only the HTTP socket transport."""
        assert actor == context.actor
        hop = backend.Hop(actor, url="http://rust.test")
        hop._http.close()
        hop._http = httpx.Client(transport=httpx.MockTransport(respond))
        return hop

    monkeypatch.setattr(rl_effects, "Hop", connect)
    return calls, log


@pytest.mark.parametrize("name", ["standard", "looped", "custom"])
def test_saved_definition_rl_updates_policy_with_original_frozen_reference(tmp_path, monkeypatch, name):
    """Real sampled action masks and policy gradients update each declared model while package publication preserves definition/reference identity."""
    saved, original = pretrained_policy(tmp_path, name)
    allow_trusted_tensor_reader(monkeypatch, saved)
    base = checkpoint.load_state(saved.directory / "model.pt", "cpu").state_dict
    context = RlContext(tmp_path / "rl")
    prep, command = rl_preparation(context, saved, original)
    calls, log = bind_parent(context, prep.request, monkeypatch)
    active = RlBackend(context)
    try:
        completed = active.handle(command, context)
        result = decode_json(context.artifacts.read_bytes(completed.result))
        assert result["iterations"] == 2 and result["signal"] == 1.0
        assert len(result["source_history"]) == 2 and all(frame.diag["policy_tokens"] > 0 for frame in context.frames)
        assert active.ready.definition == saved.manifest.definition
        assert all(parameter.grad is None and not parameter.requires_grad for parameter in active.ready.reference.parameters())
        for key, value in active.ready.reference.state_dict().items():
            torch.testing.assert_close(value, base[key], rtol=0, atol=0)
        assert any(not torch.equal(value, base[key]) for key, value in active.ready.model.state_dict().items())
        assert len(context.packages) == 3 and context.packages[-1].manifest.training_state_path is None
        metadata = TrainingPublication.model_validate_json((context.packages[-1].directory / "meta.json").read_bytes())
        assert metadata.training_kind == "rl" and metadata.config == prep.request
        assert log.pending is None and len(log.runs) == 2
        assert len([call for call in calls if call[1] == "/internal/gym/runs"]) == 2
    finally:
        active.finish_request()
        context.broker.close(timeout=2)


def execute(context, command):
    """Run and fully release the actual numerical backend, retaining only completed fixture package evidence."""
    active = RlBackend(context)
    try:
        completed = active.handle(command, context)
        return decode_json(context.artifacts.read_bytes(completed.result))
    finally:
        active.finish_request()
        context.broker.close(timeout=2)


@pytest.mark.parametrize("name", ["standard", "looped", "custom"])
def test_completed_rl_iteration_recovers_exact_optimizer_rng_and_original_reference(tmp_path, monkeypatch, name):
    """Actual saved iteration recovery matches uninterrupted weights/moments/RNG while issuing fresh gym resources only for remaining work."""
    saved, original = pretrained_policy(tmp_path, name)
    allow_trusted_tensor_reader(monkeypatch, saved)
    complete = RlContext(tmp_path / "complete")
    prep, command = rl_preparation(complete, saved, original)
    bind_parent(complete, prep.request, monkeypatch)
    expected = execute(complete, command)
    interrupted = RlContext(tmp_path / "interrupted")
    prep, command = rl_preparation(interrupted, saved, original)
    old_calls, old_log = bind_parent(interrupted, prep.request, monkeypatch)
    interrupted.cancel_at_checkpoint = True
    with pytest.raises(WorkerCancelled):
        execute(interrupted, command)
    assert len(interrupted.packages) == 1
    recovery = interrupted.packages[0]
    metadata = TrainingPublication.model_validate_json((recovery.directory / "meta.json").read_bytes())
    assert not metadata.generation.final and metadata.generation.step == 1
    resumed = RlContext(tmp_path / "resumed")
    prep, command = rl_preparation(resumed, recovery, original,
        resume=str(metadata.generation.model_id), reference=saved)
    new_calls, new_log = bind_parent(resumed, prep.request, monkeypatch)
    result = execute(resumed, command)
    expected_weights = checkpoint.load_state(complete.packages[-1].directory / "model.pt", "cpu")
    actual_weights = checkpoint.load_state(resumed.packages[-1].directory / "model.pt", "cpu")
    for key, value in expected_weights.state_dict.items():
        torch.testing.assert_close(value, actual_weights.state_dict[key], rtol=0, atol=0)
    expected_state = checkpoint.load_state(complete.packages[-2].directory / "train_state.pt", "cpu")
    actual_state = checkpoint.load_state(resumed.packages[-2].directory / "train_state.pt", "cpu")
    assert expected_state.optimizer["param_groups"] == actual_state.optimizer["param_groups"]
    for parameter, state in expected_state.optimizer["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(value, actual_state.optimizer["state"][parameter][field], rtol=0, atol=0)
    assert expected_state.extra["task_rng"] == actual_state.extra["task_rng"]
    torch.testing.assert_close(expected_state.extra["sampling_rng"], actual_state.extra["sampling_rng"], rtol=0, atol=0)
    torch.testing.assert_close(expected_state.rng["torch"], actual_state.rng["torch"], rtol=0, atol=0)
    torch.testing.assert_close(expected_state.rng["numpy_keys"], actual_state.rng["numpy_keys"], rtol=0, atol=0)
    assert {key: value for key, value in expected_state.rng.items() if key not in {"torch", "numpy_keys", "cuda"}} == {
        key: value for key, value in actual_state.rng.items() if key not in {"torch", "numpy_keys", "cuda"}}
    for expected_rng, actual_rng in zip(expected_state.rng.get("cuda", []), actual_state.rng.get("cuda", []), strict=True):
        torch.testing.assert_close(expected_rng, actual_rng, rtol=0, atol=0)
    assert result["resumed_at"] == 1 and result["source_history"] == expected["source_history"]
    assert set(old_log.episodes).isdisjoint(new_log.episodes)
    assert [call[2]["seed_base"] for call in old_calls if call[1] == "/internal/gym/runs"] == [0]
    assert [call[2]["seed_base"] for call in new_calls if call[1] == "/internal/gym/runs"] == [1]


def test_cancelled_partial_rl_iteration_has_no_recovery_generation_and_uses_fresh_runs(tmp_path, monkeypatch):
    """Stopping after part of an actual next rollout retains only the prior completed boundary and discards all current episode IDs."""
    saved, original = pretrained_policy(tmp_path, "custom")
    allow_trusted_tensor_reader(monkeypatch, saved)
    stopped = RlContext(tmp_path / "partial")
    prep, command = rl_preparation(stopped, saved, original)
    calls, old_log = bind_parent(stopped, prep.request, monkeypatch)
    stopped.stop_after_finishes = 3
    with pytest.raises(WorkerCancelled):
        execute(stopped, command)
    assert len(stopped.packages) == 1
    recovery = stopped.packages[0]
    metadata = TrainingPublication.model_validate_json((recovery.directory / "meta.json").read_bytes())
    assert metadata.generation.step == 1 and not metadata.generation.final
    assert len([call for call in calls if call[1] == "/internal/gym/runs"]) == 2
    assert len([call for call in calls if call[1].endswith("/status")]) == 1
    resumed = RlContext(tmp_path / "fresh")
    prep, command = rl_preparation(resumed, recovery, original,
        resume=str(metadata.generation.model_id), reference=saved)
    fresh_calls, fresh_log = bind_parent(resumed, prep.request, monkeypatch)
    result = execute(resumed, command)
    assert result["iterations"] == 2 and result["resumed_at"] == 1
    assert set(old_log.episodes).isdisjoint(fresh_log.episodes)
    assert set(old_log.runs).isdisjoint(fresh_log.runs)
    assert [call[2]["seed_base"] for call in fresh_calls if call[1] == "/internal/gym/runs"] == [1]


def test_rl_marker_use_cannot_authorize_later_sft_to_reset_learned_token_rows(tmp_path, monkeypatch):
    """Real RL can train markers absent from pretraining; unmeasured RL marker history must prevent subsequent SFT mean reinitialization."""
    saved, original = pretrained_policy(tmp_path, "standard", known_marker_history=True)
    allow_trusted_tensor_reader(monkeypatch, saved)
    tokenizer = byte_alphabet()
    memory = [tokenizer.token_to_id(marker) for marker in (special.MEMORY_OPEN, special.MEMORY_CLOSE)]
    before = decode_json((saved.directory / "meta.json").read_bytes())
    assert untrained_marker_ids(memory, marker_history(before, tokenizer.get_vocab_size())) == memory
    context = RlContext(tmp_path / "marker-rl")
    prep, command = rl_preparation(context, saved, original)
    bind_parent(context, prep.request, monkeypatch, prompt=f"{special.MEMORY_OPEN}one{special.MEMORY_CLOSE}")
    execute(context, command)
    after = decode_json((context.packages[-1].directory / "meta.json").read_bytes())
    initial = checkpoint.load_state(saved.directory / "model.pt", "cpu").state_dict
    tuned = checkpoint.load_state(context.packages[-1].directory / "model.pt", "cpu").state_dict
    table = next(name for name in initial if name.endswith("embedding.weight"))
    assert any(not torch.equal(initial[table][row], tuned[table][row]) for row in memory)
    assert untrained_marker_ids(memory, marker_history(after, tokenizer.get_vocab_size())) == []


def test_completed_rl_recovery_preserves_curriculum_placement_and_fixed_tests(tmp_path, monkeypatch):
    """Actual initial/periodic/final exams retain mastery, cycle and training-group position without repeating placement after recovery."""
    saved, original = pretrained_policy(tmp_path, "custom")
    allow_trusted_tensor_reader(monkeypatch, saved)
    complete = RlContext(tmp_path / "complete-curriculum")
    prep, command = rl_preparation(complete, saved, original, curriculum=True)
    bind_parent(complete, prep.request, monkeypatch)
    expected = execute(complete, command)
    interrupted = RlContext(tmp_path / "interrupted-curriculum")
    prep, command = rl_preparation(interrupted, saved, original, curriculum=True)
    bind_parent(interrupted, prep.request, monkeypatch)
    interrupted.cancel_at_checkpoint = True
    with pytest.raises(WorkerCancelled):
        execute(interrupted, command)
    recovery = interrupted.packages[0]
    metadata = TrainingPublication.model_validate_json((recovery.directory / "meta.json").read_bytes())
    resumed = RlContext(tmp_path / "resumed-curriculum")
    prep, command = rl_preparation(resumed, recovery, original, resume=str(metadata.generation.model_id),
        reference=saved, curriculum=True)
    calls, log = bind_parent(resumed, prep.request, monkeypatch)
    result = execute(resumed, command)
    assert result["curriculum"] == expected["curriculum"]
    assert result["curriculum"]["cycle"] == 3 and result["curriculum"]["training_groups"] == 2
    assert result["curriculum"]["mastery"]["math_addition_easy"]["mastered"]
    creates = [call[2] for call in calls if call[1] == "/internal/gym/runs"]
    assert [(body["split"], body["seed_base"]) for body in creates] == [("train", 1), ("validation", 20000), ("test", prep.request.holdout_base)]
    assert result["test_before"]["tasks"] == metadata.result["test_before"]["tasks"]
    assert log.pending is None and len(log.runs) == 3


@pytest.mark.parametrize("name", ["standard", "looped", "custom"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_supervised_rl_executes_saved_policy_effects_and_complete_packages(tmp_path, monkeypatch, name, device):
    """Launch the real guardian/worker RL backend and require actual parent effects, verified package adoption and cleanup."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("user-deferred CUDA hardware verification")
    saved, original = pretrained_policy(tmp_path, name)
    policy = NumericalContext(tmp_path / "policy").bootstrap.policy
    if device == "cuda":
        from llm.train import memory
        policy = policy.model_copy(update={"resources": policy.resources.model_copy(update={"device": "cuda",
            "device_allocator_bytes": memory.budget_bytes("cuda"), "process_address_bytes": 1 << 40})})
    outputs = []

    def adopt(notice, artifacts):
        """Verify and retain only complete child packages; parent inspection never loads tensors or administrator source."""
        if notice.artifact.format == "model_package_v1":
            checked = package_transfer.materialize(artifacts.verify(notice.artifact), tmp_path / f"accepted-{notice.publication_id}")
            metadata = TrainingPublication.model_validate_json((checked.directory / "meta.json").read_bytes())
            assert metadata.training_kind == "rl" and metadata.generation.step == notice.completed_step
            assert metadata.generation.final == notice.final
            outputs.append(checked)

    gate = DeviceGate()
    with WorkerManager(tmp_path / "runtime", gate) as manager, gate.admit(wait=True) as lease:
        with manager.session(lease, policy, "llm.train.rl_worker:RlBackend") as session:
            context = SimpleNamespace(directory=session.directory, artifacts=session.artifacts, request_id=None,
                actor=str(uuid4()), stop_after_finishes=None)
            prep, command = rl_preparation(context, saved, original)
            prep = prep.model_copy(update={"validation": prep.validation.model_copy(update={"request": prep.validation.request.model_copy(
                update={"profile": rl_profile(prep.request, 128, device)})})})
            command = Train.model_validate({**command.model_dump(), "config": session.artifacts.write_bytes(
                prep.model_dump_json().encode(), "training_config_v1")})
            calls, log = bind_parent(context, prep.request, monkeypatch)
            completed = session.run(command, request_id=context.request_id, effects=context.broker, adopt=adopt)
            result = decode_json(session.artifacts.read_bytes(completed.result))
        context.broker.dispatch.retire_resources()
        assert result["iterations"] == 2 and result["signal"] == 1.0
        assert len(outputs) == 3 and outputs[-1].manifest.training_state_path is None
        assert outputs[-1].manifest.definition == saved.manifest.definition
        assert log.pending is None and len(log.runs) == 2 and len(calls) >= 8
