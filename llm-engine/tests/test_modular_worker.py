"""Real in-process CPU training/package arithmetic with scoped grant/publication fixtures.

The numerical backend is exercised directly against trusted fixtures here.
This is not process containment, durable database adoption or restart evidence;
the separately supervised case retains those actual runtime requirements.
"""

import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from llm.architecture import package_transfer
from llm.architecture.package_integrity import decode_json
from llm.architecture.spec import RevisionSubject, canonical_digest
from llm.architecture.tokenizer_binding import binding_from_bytes
from llm.architecture.validation import subject_reference
from llm.architecture.parse import parse_recipe
from llm.architecture.resolve import parameter_values
from llm import storage
from llm.corpus import remote, vocab
from llm.corpus.baseline import byte_alphabet
from llm.config import TrainConfig
from llm.corpus.spans import DocSpan, Spans
from llm.orchestrate.recovery import fingerprint
from llm.runtime.artifacts import ArtifactStore
from llm.runtime.protocol import RuntimeDeadlines, RuntimePolicy, RuntimeResources, Train
from llm.runtime.worker import WorkerCancelled
from llm.runtime.supervisor import WorkerManager
from llm.model.device import DeviceGate
from llm.model import checkpoint
from llm.train.modular_records import CorpusPlan, TrainingPreparation
from llm.train.modular_records import TrainingStorage
from llm.train.modular_inputs import load_inputs
from llm.train.modular_worker import TrainingBackend, PROGRESS
from llm.train.model_preflight import training_profile
from llm.train.recompute_plan import RecomputeEstimate, TrainingRecomputation
from llm.train.plan import DataPlan
from llm.orchestrate import modular_data
from tests.test_architecture_validation import recipe_snapshot
from tests.test_architecture_builtins import build_template


class NumericalContext:
    """Own isolated real grants and inspect complete publications without claiming guardian or database behavior."""

    def __init__(self, root: Path):
        """Create finite grants/publications with production's worker-plus-one-helper process budget."""
        self.directory = root
        root.mkdir()
        self.device, self.pins = "cpu", None
        self.request_id = None
        resources = RuntimeResources(process_address_bytes=16 << 30, tree_resident_bytes=8 << 30,
            tree_memory_enforcement="preflight_and_monitor", device="cpu", device_allocator_bytes=0, owned_processes=2,
            artifact_bytes=64 << 20, total_artifact_bytes=256 << 20)
        self.bootstrap = SimpleNamespace(policy=RuntimePolicy(resources=resources))
        self.artifacts = ArtifactStore(root / "artifacts", resources)
        self.events, self.frames, self.packages = [], [], []
        self.cancelled = False

    def check_cancelled(self):
        """Expose an explicit cooperative boundary without fabricating a process cleanup certificate."""
        if self.cancelled:
            raise WorkerCancelled("fixture cancellation")

    def progress(self, event, *, force=False):
        """Retain actual typed work notifications emitted by the backend."""
        self.check_cancelled()
        self.events.append(event)

    def publish(self, artifact, *, completed_step, final=False):
        """Materialize real complete package containers or decode measured telemetry, then retire the grant."""
        self.check_cancelled()
        if artifact.format == "model_package_v1":
            checked = package_transfer.materialize(self.artifacts.verify(artifact), self.directory / f"adopted-{artifact.artifact_id}")
            metadata = decode_json((checked.directory / "meta.json").read_bytes())
            assert metadata["generation"]["step"] == completed_step
            assert metadata["generation"]["final"] == final
            self.packages.append(checked)
        else:
            assert artifact.format == "training_progress_v1"
            self.frames.append(PROGRESS.validate_json(self.artifacts.read_bytes(artifact)))
        self.artifacts.remove(artifact)


def preparation(context: NumericalContext, *, name="standard", checkpoint_target=2, device="cpu", recompute=None):
    """Prepare actual tokenizer/corpus/span bytes and a revision-bound final training profile."""
    snapshot = recipe_snapshot(name)
    reference = subject_reference(snapshot)
    cfg = TrainConfig(model_name=str(uuid4()), batch_size=2, seed=5, warmup_steps=2, compile_model=False,
        precision="bf16" if device == "cuda" else "fp32", checkpoint_target=checkpoint_target,
        recompute_blocks=recompute,
        architecture={"revision_id": reference.revision_id, "parameters": dict(snapshot.request.parameters)})
    raw = Tokenizer(WordLevel(vocab={"<|endoftext|>": 0, **{f"t{index}": index for index in range(1, 16)}},
                              unk_token="<|endoftext|>")).to_str().encode()
    binding = binding_from_bytes(uuid4(), raw)
    snapshot = snapshot.model_copy(update={"request": snapshot.request.model_copy(update={
        "subject": RevisionSubject(kind="revision", reference=reference), "tokenizer": binding,
        "parameters": parameter_values(parse_recipe(snapshot.recipe).parameters, snapshot.request.parameters),
        "context_window": 8, "profile": training_profile(cfg, 8, device)})})
    corpus = np.arange(129, dtype=np.uint16) % 16
    documents = Spans.of([DocSpan("fixture", 0, len(corpus))])
    plan = DataPlan(documents=documents, eot_id=0, shuffle=True, seed=5)
    spans_path = context.directory / "input-spans.npy"
    documents.save(spans_path)
    corpus_grant = context.artifacts.write_bytes(corpus.tobytes(), "corpus_v1")
    prep = TrainingPreparation(config=cfg, validation=snapshot,
        tokenizer=context.artifacts.write_bytes(raw, "tokenizer_v1"), data=CorpusPlan(tokens=len(corpus),
            spans=context.artifacts.stage_file(spans_path, "span_table_v1"), names=("fixture",), stages=(),
            eot_id=0, corpus_key="fixture", fingerprint=fingerprint(corpus, plan=plan), shuffle=True, seed=5))
    command = Train(type="train", job_id=snapshot.run_id, training_kind="pretrain", corpus=corpus_grant,
                    config=context.artifacts.write_bytes(prep.model_dump_json().encode(), "training_config_v1"))
    if isinstance(context, NumericalContext):
        context.request_id = snapshot.run_id
    return prep, command


def replay_allocator(name: str) -> int:
    """Derive a real captured planning ceiling where ordinary retention fails and occurrence replay fits."""
    from llm.architecture.build import build_model
    from llm.architecture.capture import CausalCapture, capture_complete
    from llm.architecture.validation import make_inputs
    from llm.train.modular_preflight import training_memory

    model = build_model(build_template(name,
        {"dimensions": 8, "heads": 2, "ff_multiplier": 2, "context_window": 8}, vocab=16).plan, device="meta")
    args = make_inputs(2, 8, 16, "meta")
    capture = capture_complete(CausalCapture(model), args)
    memory, replay = training_memory(model, capture.exported, args, allocator_bytes=2 << 30,
                                    profile_digest="a" * 64, recompute=None)
    target = (memory.estimated_bytes + min(value.estimated_bytes for value in replay.candidates)) // 2
    assert target < memory.estimated_bytes
    return (512 << 20) + target


@pytest.mark.parametrize("name", ["looped", "nested"])
def test_automatic_replay_admits_real_training_and_preserves_updates(tmp_path, monkeypatch, name):
    """Meta and real admission select the fewest fitting calls; actual updates equal unrestricted ordinary training."""
    from llm.train import modular_worker

    preflight = modular_worker.preflight
    allocator = replay_allocator(name)
    outputs = []
    for index, requested in enumerate((False, None)):
        context = NumericalContext(tmp_path / f"session-{index}")
        _prep, command = preparation(context, name=name, recompute=requested)

        def admitted(*args, **kwargs):
            """Exercise production preflight under the exact calculated ceiling without changing planner decisions."""
            if requested is None:
                kwargs["allocator_bytes"] = allocator
            return preflight(*args, **kwargs)

        monkeypatch.setattr(modular_worker, "preflight", admitted)
        backend = TrainingBackend(context)
        try:
            backend.handle(command, context)
            outputs.append({key: value.detach().clone() for key, value in backend.ready.model.state_dict().items()})
            if requested is None:
                replay = backend.ready.recomputation
                memory = backend.ready.memory
                assert 0 < len(replay.selected.occurrences) < len(backend.ready.model.recomputation_targets())
                assert memory.estimated_bytes <= memory.target_bytes < replay.candidates[0].estimated_bytes
                assert all(value.estimated_bytes > memory.target_bytes for value in replay.candidates
                           if len(value.occurrences) < len(replay.selected.occurrences))
                for checked in context.packages:
                    saved = decode_json((checked.directory / "training-plan.json").read_bytes())
                    assert saved["recomputation"]["estimator"] == "captured-occurrence-lifetimes-v1"
                    assert saved["memory"]["estimated_bytes"] == replay.selected.estimated_bytes
        finally:
            backend.finish_request()
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)


@pytest.mark.parametrize("name", ["standard", "looped"])
def test_pretrain_backend_updates_and_publishes_real_self_contained_generations(tmp_path, name, monkeypatch):
    """Real CPU updates produce valid optimizer/RNG packages, measured telemetry and one reserved finished model."""
    context = NumericalContext(tmp_path / "session")
    prep, command = preparation(context, name=name)
    backend = TrainingBackend(context)
    from llm.train import modular_worker
    preflight = modular_worker.preflight
    initial = {}

    def record_initial(*args, **kwargs):
        """Capture actual post-preflight weights while keeping the real admission/capture implementation."""
        ready = preflight(*args, **kwargs)
        initial.update({name: value.detach().clone() for name, value in ready.model.named_parameters()})
        return ready

    monkeypatch.setattr(modular_worker, "preflight", record_initial)
    try:
        result = backend.handle(command, context)
        summary = json.loads(context.artifacts.read_bytes(result.result))
        assert summary["steps"] == summary["total_steps"] == 8
        assert summary["model_id"] == prep.config.model_name
        assert summary["final_loss"] > 0
        assert any(not torch.equal(initial[name], value.detach()) for name, value in backend.ready.model.named_parameters())
        assert len(context.packages) == 3
        assert [value for value in context.events if value.unit == "batches"][-1].completed == 8
        assert context.frames and context.frames[-1].step == 8
        assert any(frame.diag and "parameter_grad_norm" in frame.diag for frame in context.frames)
        assert context.packages[-1].manifest.parameter_precision == "fp32"
        assert context.packages[-1].manifest.training_state_path is None
        saved = context.packages[-1]
        assert not storage.tokenizer_file(storage.tokenizer_build_dir(str(saved.manifest.definition.tokenizer.tokenizer_id))).exists()
        raw, binding = modular_data.tokenizer_bytes(prep.config.model_copy(update={"resume_from": prep.config.model_name}),
            None, [], None, (), saved)
        assert binding == saved.manifest.definition.tokenizer
        assert raw == (saved.directory / "tokenizer.json").read_bytes()
        for checked in context.packages[:-1]:
            assert checked.manifest.training_state_path == "train_state.pt"
            assert checked.manifest.optimizer_parameter_order
            assert len(set(checked.manifest.optimizer_parameter_order)) == len(checked.manifest.optimizer_parameter_order)
            assert "train_state.pt" in checked.integrity.files
        assert not list(context.directory.glob("generation-*"))
    finally:
        backend.finish_request()
    assert backend.ready is None and context.pins is None


def test_checkpoint_target_zero_still_publishes_the_finished_output(tmp_path):
    """Disabling recovery cadence never suppresses the reserved final model package."""
    context = NumericalContext(tmp_path / "session")
    prep, command = preparation(context, checkpoint_target=0)
    backend = TrainingBackend(context)
    try:
        backend.handle(command, context)
        assert len(context.packages) == 1
        metadata = decode_json((context.packages[0].directory / "meta.json").read_bytes())
        assert metadata["generation"]["model_id"] == prep.config.model_name
        assert metadata["generation"]["final"] is True
    finally:
        backend.finish_request()


@pytest.mark.parametrize("name,selection", [("standard", True), ("looped", 2), ("nested", True)])
def test_explicit_replay_matches_cpu_updates_and_retains_exact_policy(tmp_path, name, selection):
    """Real replay updates equal ordinary updates and every serialized generation retains profile-bound policy."""
    outputs = []
    for index, requested in enumerate((False, selection)):
        context = NumericalContext(tmp_path / f"session-{index}")
        prep, command = preparation(context, name=name, recompute=requested)
        backend = TrainingBackend(context)
        try:
            result = backend.handle(command, context)
            summary = json.loads(context.artifacts.read_bytes(result.result))
            outputs.append({key: value.detach().clone() for key, value in backend.ready.model.state_dict().items()})
            selected = backend.ready.recomputation.selected
            expected = 0 if requested is False else len(backend.ready.model.recomputation_targets()) if requested is True else requested
            assert len(selected.occurrences) == expected
            assert summary["recomputation"]["selected"]["estimated_bytes"] == backend.ready.memory.estimated_bytes
            for checked in context.packages:
                saved = decode_json((checked.directory / "training-plan.json").read_bytes())
                receipt = backend.ready.validation.report.receipt
                assert "training-plan.json" in checked.integrity.files
                assert saved["receipt_id"] == str(receipt.receipt_id)
                assert saved["definition_id"] == checked.manifest.definition_id
                assert saved["recomputation"]["selected"]["profile_digest"] == canonical_digest(receipt.identity.profile)
                assert len(saved["recomputation"]["selected"]["occurrences"]) == expected
                assert saved["memory"]["estimated_bytes"] == selected.estimated_bytes
        finally:
            backend.finish_request()
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)


@pytest.mark.parametrize("name,requested", [("standard", True), ("looped", True), ("looped", None)])
def test_saved_replay_exact_recovery_matches_final_weights_under_changed_admission(tmp_path, name, requested, monkeypatch):
    """Actual tensor/optimizer/RNG files recover a nonempty replay trajectory in a fresh trusted CPU session.

    A trusted tensor-reader fixture calls the real package/hash/state decoder;
    the production worker-marker refusal is checked before installing that seam.
    This proves saved numerical recovery, separately from guardian/process restart.
    """
    original_context = NumericalContext(tmp_path / "original")
    original, command = preparation(original_context, name=name, recompute=requested)
    from llm.train import modular_worker
    preflight = modular_worker.preflight
    allocator = replay_allocator(name) if requested is None else None

    def limited(*args, **kwargs):
        """Require actual automatic replay in the original run, before recovery receives a larger ceiling."""
        if allocator is not None:
            kwargs["allocator_bytes"] = allocator
        return preflight(*args, **kwargs)

    monkeypatch.setattr(modular_worker, "preflight", limited)
    backend = TrainingBackend(original_context)
    try:
        backend.handle(command, original_context)
        expected = {key: value.detach().clone() for key, value in backend.ready.model.state_dict().items()}
        selected = backend.ready.recomputation.selected
        assert selected.occurrences
    finally:
        backend.finish_request()
    saved = original_context.packages[0]
    with pytest.raises(ValueError, match="admitted worker"):
        checkpoint.load_state(saved.directory / "train_state.pt")

    def trusted_saved_state(path, device):
        """Decode only this trusted fixture's actual files without spoofing a runtime-child environment marker."""
        blob = torch.load(path, map_location=device, weights_only=True)
        return checkpoint._load_modular_state(path, blob, device)

    monkeypatch.setattr(checkpoint, "load_state", trusted_saved_state)
    monkeypatch.setattr(modular_worker, "preflight", preflight)
    context = NumericalContext(tmp_path / "recovered")
    prep, command = preparation(context, name=name, recompute=requested)
    cfg = original.config.model_copy(update={"model_name": str(uuid4()),
        "resume_from": decode_json((saved.directory / "meta.json").read_bytes())["generation"]["model_id"],
        "recover_exact": True, "architecture": None})
    snapshot = original.validation.model_copy(update={"run_id": prep.validation.run_id})
    prep = prep.model_copy(update={"config": cfg, "validation": snapshot})
    archive = context.directory / "recovery.tar"
    package_transfer.write_archive(saved.directory, archive)
    command = Train.model_validate({**command.model_dump(),
        "config": context.artifacts.write_bytes(prep.model_dump_json().encode(), "training_config_v1"),
        "recovery": context.artifacts.stage_file(archive, "model_package_v1")})
    backend = TrainingBackend(context)
    try:
        backend.handle(command, context)
        assert backend.ready.recomputation.selected == selected
        for key, value in backend.ready.model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        assert context.frames[-1].step == 8
        assert context.frames[0].step > decode_json((saved.directory / "meta.json").read_bytes())["generation"]["step"]
    finally:
        backend.finish_request()


def test_bad_corpus_binding_never_reaches_model_or_optimizer(tmp_path, monkeypatch):
    """A corpus ID outside finalized vocabulary fails before source/model construction and AdamW."""
    context = NumericalContext(tmp_path / "session")
    _prep, command = preparation(context)
    wrong = context.artifacts.write_bytes(np.full(129, 16, dtype=np.uint16).tobytes(), "corpus_v1")
    command = command.model_copy(update={"corpus": wrong})

    def forbidden(*args, **kwargs):
        """Detect premature source/model/optimizer work before input verification."""
        pytest.fail("invalid data reached numerical construction")

    monkeypatch.setattr("llm.train.modular_worker.preflight", forbidden)
    monkeypatch.setattr("torch.optim.AdamW", forbidden)
    backend = TrainingBackend(context)
    with pytest.raises(ValueError, match="outside the final vocabulary"):
        backend.handle(command, context)
    backend.finish_request()
    assert context.packages == []


def test_unknown_replay_count_fails_before_optimizer_or_publication(tmp_path, monkeypatch):
    """An integer policy cannot silently clamp to an unavailable occurrence or begin an update."""
    context = NumericalContext(tmp_path / "session")
    _prep, command = preparation(context, recompute=999)

    def forbidden(*args, **kwargs):
        """Detect optimizer allocation after a replay policy should already have failed admission."""
        pytest.fail("unknown replay count reached AdamW")

    monkeypatch.setattr("torch.optim.AdamW", forbidden)
    backend = TrainingBackend(context)
    try:
        with pytest.raises(ValueError, match="count"):
            backend.handle(command, context)
        assert context.packages == []
    finally:
        backend.finish_request()


def test_compiled_cuda_replay_refuses_before_numerical_construction(tmp_path, monkeypatch):
    """Check the unsupported execution-policy boundary using CPU grants, without claiming CUDA arithmetic."""
    context = NumericalContext(tmp_path / "session")
    prep, command = preparation(context, recompute=True)
    cfg = prep.config.model_copy(update={"compile_model": True})
    profile = training_profile(cfg, 8, "cuda")
    snapshot = prep.validation.model_copy(update={"request": prep.validation.request.model_copy(update={"profile": profile})})
    prep = prep.model_copy(update={"config": cfg, "validation": snapshot})
    command = command.model_copy(update={"config": context.artifacts.write_bytes(prep.model_dump_json().encode(),
        "training_config_v1")})
    context.device = "cuda"

    def forbidden(*args, **kwargs):
        """Detect CUDA/model/optimizer construction after the unsupported policy should have been rejected."""
        pytest.fail("compiled replay reached numerical construction")

    monkeypatch.setattr("llm.train.modular_worker.preflight", forbidden)
    monkeypatch.setattr("torch.optim.AdamW", forbidden)
    backend = TrainingBackend(context)
    with pytest.raises(ValueError, match="compile_model=false"):
        backend.handle(command, context)
    assert context.packages == []


def test_exact_recovery_requires_the_retained_replay_profile_and_selection(tmp_path):
    """The continuation decision refuses missing/stale replay authority even when data/config/cursor agree."""
    context = NumericalContext(tmp_path / "session")
    prep, command = preparation(context)
    inputs = load_inputs(command, prep, context)
    original = prep.config
    cfg = original.model_copy(update={"recover_exact": True, "resume_from": str(uuid4()), "architecture": None})
    prep = prep.model_copy(update={"config": cfg})
    selected = RecomputeEstimate("a" * 64, (), 100)
    replay = TrainingRecomputation(selected, (selected,))
    state = checkpoint.TrainState(arch={}, state_dict={}, step=4, total_steps=8, extra={
        "config": prep.stated_config(original), "cursor": {"epoch": 0}, "data_fingerprint": prep.data.fingerprint})
    backend = TrainingBackend(context)
    backend.ready = SimpleNamespace(state=state, recomputation=replay)
    effective = cfg.model_copy(update={"ctx_window": prep.validation.request.context_window})
    with pytest.raises(ValueError, match="exact recovery"):
        backend.continuation(prep, inputs, effective)
    state.extra["recomputation_policy"] = replay.as_dict()["selected"]
    assert backend.continuation(prep, inputs, effective) is state
    state.extra["recomputation_policy"]["profile_digest"] = "b" * 64
    with pytest.raises(ValueError, match="exact recovery"):
        backend.continuation(prep, inputs, effective)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("recompute", [False, True])
def test_supervised_pretrain_executes_the_real_backend_and_materializes_packages(tmp_path, device, recompute):
    """Use actual guardian/publication ownership with production's worker-plus-one-framework-helper process bound."""
    gate = DeviceGate()
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("user-deferred CUDA hardware verification")
    from llm.train import memory

    resources = RuntimeResources(process_address_bytes=1 << 40 if device == "cuda" else 16 << 30, tree_resident_bytes=8 << 30,
        tree_memory_enforcement="preflight_and_monitor", device=device, owned_processes=2,
        device_allocator_bytes=memory.budget_bytes("cuda") if device == "cuda" else 0,
        artifact_bytes=64 << 20, total_artifact_bytes=256 << 20)
    policy = RuntimePolicy(resources=resources, deadlines=RuntimeDeadlines(validation_seconds=180.0))
    adopted = []

    def adopt(notice, artifacts):
        """Verify actual output grants and preserve complete parent-owned packages before acknowledging publication."""
        if notice.artifact.format == "model_package_v1":
            checked = package_transfer.materialize(artifacts.verify(notice.artifact), tmp_path / f"package-{notice.publication_id}")
            saved_plan = decode_json((checked.directory / "training-plan.json").read_bytes())
            assert bool(saved_plan["recomputation"]["selected"]["occurrences"]) is recompute
            adopted.append(checked.manifest)
        else:
            assert notice.artifact.format == "training_progress_v1"
            PROGRESS.validate_json(artifacts.read_bytes(notice.artifact))

    with WorkerManager(tmp_path / "runtime", gate) as manager, gate.admit() as lease:
        with manager.session(lease, policy, "llm.train.modular_worker:TrainingBackend") as session:
            prep, command = preparation(session, device=device, recompute=recompute)
            result = session.run(command, request_id=prep.validation.run_id, adopt=adopt)
            summary = json.loads(session.artifacts.read_bytes(result.result))
            assert summary["steps"] == 8 and summary["model_id"] == prep.config.model_name
        assert len(adopted) == 3
        assert all(value.parameter_precision == "fp32" for value in adopted)
    with gate.admit() as available:
        available.require_active()


def test_failed_capture_never_constructs_an_optimizer(tmp_path, monkeypatch):
    """A real backend preflight failure has no optimizer fallback or package publication."""
    from llm.architecture.errors import ArchitectureError

    context = NumericalContext(tmp_path / "session")
    _prep, command = preparation(context)

    def failed_capture(*args, **kwargs):
        """Inject the strict capture refusal at the actual preflight boundary."""
        raise ArchitectureError("capture_failed", "fixture strict capture refusal")

    def forbidden(*args, **kwargs):
        """Detect any optimizer construction following a failed strict capture."""
        pytest.fail("failed capture reached AdamW")

    monkeypatch.setattr("llm.train.modular_preflight.capture_complete", failed_capture)
    monkeypatch.setattr("torch.optim.AdamW", forbidden)
    backend = TrainingBackend(context)
    with pytest.raises(ArchitectureError, match="strict capture refusal"):
        backend.handle(command, context)
    backend.finish_request()
    assert context.packages == []


def test_parent_preparation_preserves_actor_and_recipe_context_without_source_execution(tmp_path, data_root, monkeypatch):
    """The real pack/measurement path uses caller-scoped data and authoritative recipe context before worker launch."""
    raw = byte_alphabet().to_str().encode()
    binding = binding_from_bytes(uuid4(), raw)
    directory = storage.tokenizer_build_dir(str(binding.tokenizer_id))
    directory.mkdir(parents=True)
    storage.tokenizer_file(directory).write_bytes(raw)
    storage.tokenizer_meta_path(str(binding.tokenizer_id)).write_text(json.dumps({
        "vocab_size": binding.vocabulary_size, "fingerprint": binding.fingerprint[:vocab.FINGERPRINT_CHARS]}))
    opened, closed, frames = [], [], []
    dataset_id = str(uuid4())

    class DataClient:
        """Own deterministic authorized dataset pages without claiming a real service transport."""

        def __init__(self, actor):
            """Record the initiating actor handed to the production parent acquisition seam."""
            opened.append(actor)

        def __enter__(self):
            """Expose this scoped data connection to packing."""
            return self

        def __exit__(self, *args):
            """Record closure before any numerical worker can start."""
            closed.append(True)

        def dataset(self, identity):
            """Resolve the one explicitly selected dataset with stable byte/count/version facts."""
            assert identity == dataset_id
            return remote.Dataset(identity, "owned text", "pretrain", 4, 80, "fixed-version")

        def documents(self, identity, *, cap=None):
            """Yield actual prose to the real tokenizer/packer without retaining a text corpus."""
            assert identity == dataset_id
            yield from ({"content": "one two three four five"} for _ in range(cap or 4))

    def forbidden(*args, **kwargs):
        """Reject custom source/model construction while the parent is acquiring and measuring data."""
        pytest.fail("parent data preparation executed source")

    monkeypatch.setattr(modular_data.remote, "Client", DataClient)
    monkeypatch.setattr("llm.architecture.source.SourceLoader.load", forbidden)
    cfg = TrainConfig(model_name=str(uuid4()), tokenizer_id=str(binding.tokenizer_id), datasets=[dataset_id],
                      architecture={"revision_id": str(uuid4()), "parameters": {}})
    prepared = modular_data.prepare(cfg, context_window=8, reserved_names=(), directory=tmp_path / "prepared",
                                    actor="initiating-actor", progress=frames.append, cancel=Event())
    assert opened == ["initiating-actor"] and closed == [True]
    assert cfg.ctx_window == 256 and frames[-1].ctx_window == 8
    assert prepared.binding == binding and prepared.tokens > 16
    assert prepared.spans.is_file() and prepared.tokenizer.read_bytes() == raw


@pytest.mark.parametrize("available,expected", [(True, "bf16"), (False, "fp16")])
def test_effective_cuda_precision_records_the_actual_bf16_fallback(monkeypatch, available, expected):
    """CPU policy mocks record the loop's effective arithmetic; they do not verify CUDA execution."""
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: available)
    profile = training_profile(TrainConfig(model_name="fixture", precision="bf16"), 8, "cuda")
    assert profile.precision == expected


@pytest.mark.parametrize("field", ["revision_id", "parameters"])
def test_frozen_preparation_rejects_a_different_submitted_selection(tmp_path, field):
    """Typed preparation cannot bind one request's recipe values to another frozen source configuration."""
    context = NumericalContext(tmp_path / "session")
    prep, _command = preparation(context)
    value = json.loads(prep.model_dump_json())
    value["config"]["architecture"][field] = str(uuid4()) if field == "revision_id" else {"dimensions": 9}
    with pytest.raises(ValueError, match="differ from the submitted selection"):
        TrainingPreparation.model_validate(value)


@pytest.mark.parametrize("refusal", ["quota", "free_space"])
def test_checkpoint_admission_refuses_before_adamw_or_updates(tmp_path, monkeypatch, refusal):
    """Actual numerical preflight cannot enter optimization without its serialized-generation quota and staging headroom."""
    context = NumericalContext(tmp_path / "session")
    prep, command = preparation(context)
    if refusal == "quota":
        prep = prep.model_copy(update={"storage": TrainingStorage(package_bytes=1024, file_bytes=1024)})
        command = command.model_copy(update={"config": context.artifacts.write_bytes(prep.model_dump_json().encode(), "training_config_v1")})
    else:
        monkeypatch.setattr("llm.train.modular_checkpoints.shutil.disk_usage", lambda path: SimpleNamespace(free=0))

    def forbidden(*args, **kwargs):
        """Detect optimizer construction after an ineffective checkpoint storage admission."""
        pytest.fail("storage refusal reached AdamW")

    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    backend = TrainingBackend(context)
    try:
        with pytest.raises(ValueError, match="checkpoint byte budget|insufficient free space"):
            backend.handle(command, context)
        assert context.packages == []
    finally:
        backend.finish_request()
