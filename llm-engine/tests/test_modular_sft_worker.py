"""Actual saved-package SFT numerical execution, separate from supervision and durable parent adoption."""

import copy
import hashlib
import json
from uuid import uuid4

import numpy as np
import pytest
import torch
import yaml

from llm.architecture import package_transfer
from llm.architecture.package_integrity import decode_json
from llm.architecture.spec import RevisionRef, RevisionSubject
from llm.architecture.validation_records import PinnedSource
from llm.architecture.wire import ArchitectureSelection
from llm.config import SftConfig
from llm.model import checkpoint
from llm.runtime.protocol import Train
from llm.runtime.worker import WorkerCancelled
from llm.train import data_identity
from llm.train.model_preflight import sft_loop_config, training_profile
from llm.train.modular_records import CorpusPlan, SftDataset, SftInputs, TrainingPreparation
from llm.train.modular_worker import TrainingBackend
from llm.train.plan import DataPlan
from llm.train import modular_preflight, sft_diagnostics
from llm.train.memory import planning_target
from llm.architecture.errors import ArchitectureError
from llm.corpus.spans import DocSpan, Spans
from tests.test_architecture_core import revision
from tests.test_modular_worker import NumericalContext, preparation
from tests.test_token_rows import ROW_SOURCE, row_recipe


def base_package(tmp_path, name, *, known_markers=False):
    """Train and publish genuine base weights with exact source/tokenizer/capture artifacts."""
    context = NumericalContext(tmp_path / "base")
    prep, command = preparation(context, name="standard" if name == "custom" else name, checkpoint_target=0)
    if known_markers:
        prep = TrainingPreparation.model_validate({**prep.model_dump(), "data": {
            **prep.data.model_dump(), "special_ids_seen": ()}})
        command = Train.model_validate({**command.model_dump(), "config": context.artifacts.write_bytes(
            prep.model_dump_json().encode(), "training_config_v1")})
    if name == "custom":
        source = ROW_SOURCE.split("    def initialize_token_rows")[0].replace('capabilities=("token_rows",)', 'capabilities=()')
        source = source.replace("self.readout = nn.Linear(width, vocab, bias=False)",
            "self.readout = nn.Linear(width, vocab, bias=False)\n        self.noise = nn.Dropout(0.25)")
        source = source.replace("self.readout(self.lookup(tokens))", "self.readout(self.noise(self.lookup(tokens)))")
        block = revision(source)
        recipe = row_recipe(block.reference).model_copy(update={"context_window": 8})
        raw = yaml.safe_dump(recipe.model_dump(mode="json", exclude_none=True))
        reference = RevisionRef(kind="architecture", resource_id=uuid4(), revision_id=uuid4(),
                                content_hash=hashlib.sha256(raw.encode()).hexdigest())
        request = prep.validation.request.model_copy(update={"subject": RevisionSubject(kind="revision", reference=reference),
                                                             "parameters": {}})
        snapshot = prep.validation.model_copy(update={"recipe": raw, "request": request,
            "dependencies": (PinnedSource(reference=block.reference, source=source),), "builtins": ()})
        cfg = prep.config.model_copy(update={"architecture": ArchitectureSelection(revision_id=reference.revision_id, parameters={})})
        prep = TrainingPreparation.model_validate({**prep.model_dump(), "config": cfg, "validation": snapshot})
        command = Train.model_validate({**command.model_dump(), "config": context.artifacts.write_bytes(
            prep.model_dump_json().encode(), "training_config_v1")})
    backend = TrainingBackend(context)
    try:
        backend.handle(command, context)
    finally:
        backend.finish_request()
    return context.packages[-1], prep


def test_saved_pretrain_marker_history_authorizes_only_new_sft_rows(tmp_path, monkeypatch):
    """A package produced by pretraining preserves complete known marker history for capability-owned SFT initialization."""
    saved, original = base_package(tmp_path, "standard", known_markers=True)
    metadata = decode_json((saved.directory / "meta.json").read_bytes())
    assert metadata["result"]["dataset"] == {"special_ids_seen": [], "marker_history_complete": True}
    allow_trusted_tensor_reader(monkeypatch, saved)
    context = NumericalContext(tmp_path / "known-marker-sft")
    prep, command = sft_preparation(context, saved, original)
    prep = TrainingPreparation.model_validate({**prep.model_dump(), "sft": {**prep.sft.model_dump(),
        "dataset": {**prep.sft.dataset.model_dump(), "special_ids_seen": (1, 2)}}})
    command = Train.model_validate({**command.model_dump(), "config": context.artifacts.write_bytes(
        prep.model_dump_json().encode(), "training_config_v1")})
    backend = TrainingBackend(context)
    try:
        result = backend.handle(command, context)
        summary = decode_json(context.artifacts.read_bytes(result.result))
        assert summary["warm_started"] == 2
        assert summary["warm_start"] == {"status": "initialized", "initialized_ids": [1, 2], "skipped_ids": []}
        assert summary["dataset"]["special_ids_seen"] == [1, 2] and summary["dataset"]["marker_history_complete"]
    finally:
        backend.finish_request()


def allow_trusted_tensor_reader(monkeypatch, saved):
    """Check production parent refusal, then use the genuine decoder only for this trusted numerical test's own packages."""
    with pytest.raises(ValueError, match="admitted worker"):
        checkpoint.load_state(saved.directory / "model.pt")

    def trusted(path, device):
        """Read safe tensor data without spoofing runtime-child identity or process admission."""
        blob = torch.load(path, map_location=device, weights_only=True)
        return checkpoint._load_modular_state(path, blob, device)

    monkeypatch.setattr(checkpoint, "load_state", trusted)


def sft_preparation(context, saved, original, *, resume=None, measure=False, reference=None):
    """Stage genuine numeric corpus/mask/span and inherited package grants for one validated SFT command."""
    base_id = original.config.model_name
    request = SftConfig(model_name=str(uuid4()), base_model=base_id, resume_from=resume,
        epochs=1, batch_size=2, warmup_steps=0, lr=0.001, seed=19, checkpoint_target=4,
        precision="fp32", compile_model=False, measure_base_kl=measure)
    cfg = sft_loop_config(request, 8)
    context.request_id = uuid4()
    validation = original.validation.model_copy(update={"run_id": context.request_id,
        "request": original.validation.request.model_copy(update={"profile": training_profile(cfg, 8, "cpu")})})
    corpus = np.arange(129, dtype=np.uint16) % 16
    mask = np.ones(len(corpus), dtype=np.uint8)
    mask[:32] = 0
    documents = Spans.of([DocSpan("sft", 0, len(corpus))])
    plan = DataPlan(documents=documents, eot_id=0, shuffle=False, seed=request.seed)
    request = request.model_copy(update={"shuffle": False})
    cfg = sft_loop_config(request, 8)
    span_path = context.directory / "sft-spans.npy"
    documents.save(span_path)
    raw = (saved.directory / "tokenizer.json").read_bytes()
    dataset = SftDataset(sets=(), reasoning="keep", conversations=1, total_tokens=len(corpus),
                         trainable_tokens=int(mask.sum()), skipped=0, special_ids_seen=())
    sft = SftInputs(request=request, mask=context.artifacts.write_bytes(mask.tobytes(), "loss_mask_v1"), dataset=dataset)
    grants = {}
    if measure:
        for name, size in (("pretrain_probe", 9), ("chat_probe", 8)):
            path = context.directory / f"{name}.npy"
            np.save(path, (np.arange(2 * size, dtype=np.int64) % 16).reshape(2, size), allow_pickle=False)
            grants[name] = context.artifacts.stage_file(path, "sft_probe_v1")
    if reference is not None:
        path = context.directory / "reference.tar"
        package_transfer.write_archive(reference.directory, path)
        grants["reference"] = context.artifacts.stage_file(path, "model_package_v1")
    sft = SftInputs.model_validate({**sft.model_dump(), **grants})
    prep = TrainingPreparation(config=cfg, validation=validation,
        tokenizer=context.artifacts.write_bytes(raw, "tokenizer_v1"),
        data=CorpusPlan(tokens=len(corpus), spans=context.artifacts.stage_file(span_path, "span_table_v1"),
            names=("sft",), stages=(), eot_id=0, corpus_key="sft-fixture",
            fingerprint=data_identity.fingerprint(corpus, mask, plan=plan), shuffle=False, seed=request.seed), sft=sft)
    archive = context.directory / "source.tar"
    package_transfer.write_archive(saved.directory, archive)
    command = Train(type="train", job_id=context.request_id, training_kind="sft",
        config=context.artifacts.write_bytes(prep.model_dump_json().encode(), "training_config_v1"),
        corpus=context.artifacts.write_bytes(corpus.tobytes(), "corpus_v1"),
        recovery=context.artifacts.stage_file(archive, "model_package_v1"))
    return prep, Train.model_validate_json(command.model_dump_json())


@pytest.mark.parametrize("name", ["standard", "looped", "custom"])
def test_saved_definition_sft_updates_and_publishes_real_masked_generations(tmp_path, monkeypatch, name):
    """Inherited standard/shared/custom models train with exact masks and publish typed SFT checkpoints and output."""
    saved, original = base_package(tmp_path, name)
    allow_trusted_tensor_reader(monkeypatch, saved)
    before = checkpoint.load_state(saved.directory / "model.pt", "cpu").state_dict
    context = NumericalContext(tmp_path / "sft")
    prep, command = sft_preparation(context, saved, original)
    backend = TrainingBackend(context)
    try:
        result = backend.handle(command, context)
        summary = json.loads(context.artifacts.read_bytes(result.result))
        assert summary["steps"] == 6 and summary["total_steps"] == 8
        assert summary["base_model"] == original.config.model_name
        assert summary["dataset"]["trainable_tokens"] == 97
        assert summary["warm_started"] == 0
        assert summary["warm_start"]["status"] == ("initialized" if name == "standard" else "skipped_unsupported")
        assert any(not torch.equal(before[key], value) for key, value in backend.ready.model.state_dict().items())
        assert backend.ready.definition == saved.manifest.definition
        for package in context.packages:
            metadata = decode_json((package.directory / "meta.json").read_bytes())
            assert metadata["training_kind"] == "sft" and metadata["config"]["base_model"] == original.config.model_name
            assert (package.directory / "tokenizer.json").read_bytes() == (saved.directory / "tokenizer.json").read_bytes()
            assert "architecture" not in metadata["config"] and "dimensions" not in metadata["config"]
            if metadata["generation"]["final"]:
                assert metadata["generation"]["skipped_batches"] == 2
    finally:
        backend.finish_request()
    assert backend.ready is None and context.pins is None


def test_sft_reference_is_frozen_admitted_and_reports_real_probe_metrics(tmp_path, monkeypatch):
    """Actual frozen-base construction is charged and optional mask/probe telemetry cannot alter the reference."""
    saved, original = base_package(tmp_path, "custom")
    allow_trusted_tensor_reader(monkeypatch, saved)
    initial = checkpoint.load_state(saved.directory / "model.pt", "cpu").state_dict
    states, random_states = [], []
    for index, measuring in enumerate((False, True)):
        context = NumericalContext(tmp_path / f"sft-{index}")
        _prep, command = sft_preparation(context, saved, original, measure=measuring)
        backend = TrainingBackend(context)
        try:
            backend.handle(command, context)
            states.append(copy.deepcopy(backend.ready.model.state_dict()))
            random_states.append(torch.get_rng_state().clone())
            if measuring:
                assert backend.ready.reference is not None and backend.ready.memory.reference_bytes > 0
                assert all(not value.requires_grad for value in backend.ready.reference.parameters())
                for key, value in backend.ready.reference.state_dict().items():
                    torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
                measured = [frame.diag for frame in context.frames if frame.diag]
                assert measured and all({"loss_masked", "pretrain_loss", "base_kl"} <= item.keys() for item in measured)
                assert all("loss_unmasked" not in item and item["base_kl"] >= -1e-6 for item in measured)
        finally:
            backend.finish_request()
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
    assert torch.equal(random_states[0], random_states[1])


@pytest.mark.parametrize("name", ["standard", "looped", "custom"])
def test_saved_sft_cursor_optimizer_rng_and_original_reference_recover_exactly(tmp_path, monkeypatch, name):
    """Real serialized SFT recovery resumes after targetless batches; custom dropout and frozen-base identity survive."""
    saved, original = base_package(tmp_path, name)
    allow_trusted_tensor_reader(monkeypatch, saved)
    original_context = NumericalContext(tmp_path / "original-sft")
    _prep, command = sft_preparation(original_context, saved, original, measure=name == "custom")
    backend = TrainingBackend(original_context)
    try:
        backend.handle(command, original_context)
        expected = copy.deepcopy(backend.ready.model.state_dict())
        wanted_rng = torch.get_rng_state().clone()
    finally:
        backend.finish_request()
    partial = NumericalContext(tmp_path / "interrupted-sft")
    _prep, interrupted_command = sft_preparation(partial, saved, original, measure=name == "custom")
    publish = partial.publish

    def cancel_after_complete_package(artifact, *, completed_step, final=False):
        """Keep the actual first complete numerical recovery package, then stop before another batch or final output."""
        publish(artifact, completed_step=completed_step, final=final)
        if artifact.format == "model_package_v1" and not final:
            partial.cancelled = True
            partial.check_cancelled()

    partial.publish = cancel_after_complete_package
    interrupted = TrainingBackend(partial)
    try:
        with pytest.raises(WorkerCancelled):
            interrupted.handle(interrupted_command, partial)
        assert len(partial.packages) == 1
    finally:
        interrupted.finish_request()
    point = partial.packages[0]
    metadata = decode_json((point.directory / "meta.json").read_bytes())
    context = NumericalContext(tmp_path / "recovered-sft")
    _prep, command = sft_preparation(context, point, original, resume=metadata["generation"]["model_id"],
        measure=name == "custom", reference=saved if name == "custom" else None)
    backend = TrainingBackend(context)
    try:
        backend.handle(command, context)
        for key, value in backend.ready.model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        assert torch.equal(wanted_rng, torch.get_rng_state())
        assert context.frames[0].step > metadata["generation"]["step"]
        if name == "custom":
            base = checkpoint.load_state(saved.directory / "model.pt", "cpu").state_dict
            for key, value in backend.ready.reference.state_dict().items():
                torch.testing.assert_close(value, base[key], rtol=0, atol=0)
    finally:
        backend.finish_request()


@pytest.mark.parametrize("invalid", ["nonbinary", "zero", "count", "fingerprint", "probe", "source"])
def test_invalid_sft_inputs_refuse_before_source_construction(tmp_path, monkeypatch, invalid):
    """Malformed actual masks/probes, stale identity and foreign saved models never reach model construction or optimization."""
    saved, original = base_package(tmp_path, "standard")
    context = NumericalContext(tmp_path / "invalid-sft")
    prep, command = sft_preparation(context, saved, original)
    values = prep.model_dump()
    if invalid in {"nonbinary", "zero", "count", "fingerprint"}:
        mask = bytearray(context.artifacts.read_bytes(prep.sft.mask))
        if invalid == "nonbinary":
            mask[40] = 2
        elif invalid == "zero":
            mask = bytearray(len(mask))
            values["sft"]["dataset"]["trainable_tokens"] = 0
        elif invalid == "count":
            values["sft"]["dataset"]["trainable_tokens"] = 96
        else:
            mask[30], mask[40] = mask[40], mask[30]
        values["sft"]["mask"] = context.artifacts.write_bytes(bytes(mask), "loss_mask_v1")
    elif invalid == "probe":
        path = context.directory / "bad-probe.npy"
        np.save(path, np.full((1, 9), 16, dtype=np.int64), allow_pickle=False)
        values["sft"]["pretrain_probe"] = context.artifacts.stage_file(path, "sft_probe_v1")
    else:
        request = prep.sft.request.model_copy(update={"base_model": str(uuid4())})
        values["sft"]["request"] = request
        values["config"] = sft_loop_config(request, 8)
    prep = TrainingPreparation.model_validate(values)
    command = Train.model_validate({**command.model_dump(), "config": context.artifacts.write_bytes(
        prep.model_dump_json().encode(), "training_config_v1")})

    def forbidden(*args, **kwargs):
        """Detect any source loading or weight construction after invalid external data."""
        pytest.fail("invalid SFT inputs reached source/model work")

    monkeypatch.setattr(modular_preflight, "build_model", forbidden)
    monkeypatch.setattr("llm.architecture.source.SourceLoader.load", forbidden)
    backend = TrainingBackend(context)
    try:
        with pytest.raises(ValueError):
            backend.handle(command, context)
        assert not context.frames and not context.packages and context.pins is None
    finally:
        backend.finish_request()


def test_sft_reference_budget_refuses_before_any_real_model_allocation(tmp_path, monkeypatch):
    """A cap that fits training but excludes its actual reference estimate refuses during meta-only admission."""
    _saved, original = base_package(tmp_path, "standard")
    ready = modular_preflight.preflight(original.validation, allocator_bytes=2 << 30, progress=lambda _: None,
        recompute=False, restore_reference=lambda _model, _definition: None)
    try:
        estimate = ready.memory.estimated_bytes
        target = estimate - ready.memory.reference_bytes // 2
        assert estimate - ready.memory.reference_bytes < target < estimate
    finally:
        ready.close()
    low, high = 1, 2 << 30
    while low < high:
        middle = (low + high) // 2
        if planning_target(middle) < target:
            low = middle + 1
        else:
            high = middle
    original_build = modular_preflight.build_model
    devices = []

    def meta_only(*args, **kwargs):
        """Record actual allocations and fail if a rejected reference reaches real weights."""
        devices.append(kwargs["device"])
        assert kwargs["device"] == "meta"
        return original_build(*args, **kwargs)

    monkeypatch.setattr(modular_preflight, "build_model", meta_only)
    with pytest.raises(ArchitectureError, match="fit|budget|target"):
        modular_preflight.preflight(original.validation, allocator_bytes=low, progress=lambda _: None,
            recompute=False, restore_reference=lambda _model, _definition: None)
    assert devices == ["meta", "meta"]


def test_sft_optional_reference_oom_preserves_actual_updates_and_rng(tmp_path, monkeypatch):
    """Only optional KL diagnostics disable after allocation failure; training and frozen reference remain intact."""
    saved, original = base_package(tmp_path, "custom")
    allow_trusted_tensor_reader(monkeypatch, saved)
    states, rngs = [], []
    calls = []

    def diagnostic_oom(*args):
        """Simulate a diagnostic allocator failure inside preserved eval/RNG contexts."""
        calls.append(True)
        torch.rand(3)
        raise torch.cuda.OutOfMemoryError("optional SFT KL allocation")

    for failed in (False, True):
        context = NumericalContext(tmp_path / f"optional-{failed}")
        _prep, command = sft_preparation(context, saved, original, measure=failed)
        backend = TrainingBackend(context)
        with monkeypatch.context() as patch:
            if failed:
                patch.setattr(sft_diagnostics, "reference_kl", diagnostic_oom)
            try:
                backend.handle(command, context)
                states.append(copy.deepcopy(backend.ready.model.state_dict()))
                rngs.append(torch.get_rng_state().clone())
                assert all(not frame.diag or "base_kl" not in frame.diag for frame in context.frames)
            finally:
                backend.finish_request()
    assert calls == [True]
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
    assert torch.equal(rngs[0], rngs[1])
