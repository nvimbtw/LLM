"""Actual saved-model serving arithmetic through explicit trusted numerical seams; process containment is tested separately."""

from uuid import uuid4

import pytest
import torch

from llm.architecture import package_transfer
from llm.architecture.package_execution import construct_package, load_granted_package
from llm.architecture.package_integrity import encode_json
from llm.architecture.spec import ResolvedDefinition, canonical_digest
from llm.architecture.wire import AttentionMatrix, AttentionSelection, InspectionCatalog
from llm.model.generation import prepare_prompt
from llm.config import ChatConfig, ChatTurn, EvalConfig
from llm.corpus import bpe
from llm.orchestrate import inference_worker
from llm.model import likelihood
from llm.model.inference_records import AttentionInput, InspectionInput, GenerationInput, GenerationResult, MeasurementInput, MeasurementResult
from llm.orchestrate.inference_worker import InferenceBackend
from llm.runtime.protocol import Generate, Load, Measure, RuntimePins, Tokens
from llm.runtime.worker import WorkerCancelled
from tests.test_modular_rl_worker import pretrained_policy
from tests.test_modular_sft_worker import allow_trusted_tensor_reader
from tests.test_modular_worker import NumericalContext


class InferenceContext(NumericalContext):
    """Keep genuine finite grants and cancellation without spoofing a numerical child or guardian certificate."""

    def __init__(self, root):
        """Give the explicit numerical seam a stable test identity without any process or guardian authorization."""
        super().__init__(root)
        self.bootstrap.worker_id = uuid4()

    def emit(self, event):
        """Collect bounded runtime events emitted by actual generation code."""
        self.check_cancelled()
        self.events.append(event)


def stage_load(context, saved):
    """Snapshot a complete actual trained package into the same validated Load grants used in production."""
    receipt = saved.receipts[0]
    pins = RuntimePins(definition_id=saved.manifest.definition_id, bindings_digest=receipt.identity.parameter_digest,
        profile_digest=canonical_digest(receipt.identity.profile), receipt_id=receipt.receipt_id)
    archive = context.directory / "input.tar"
    package_transfer.write_archive(saved.directory, archive)
    return Load(type="load", pins=pins, definition=context.artifacts.write_bytes(encode_json(saved.manifest.definition), "definition_v1"),
        package=context.artifacts.stage_file(archive, "model_package_v1"))


def trusted_package_loader(monkeypatch, saved):
    """Verify real parent loading is denied, then expose shared construction only for this numerical test's own validated private archive."""
    allow_trusted_tensor_reader(monkeypatch, saved)

    def trusted(command, context, *, preflight):
        """Use real archive/definition/pin/source/tensor logic without claiming OS admission or setting a runtime-child marker."""
        with pytest.raises(ValueError, match="admitted numerical worker"):
            load_granted_package(command, context, preflight=preflight)
        checked = package_transfer.materialize(context.artifacts.verify(command.package), context.directory / f"model-{command.package.artifact_id}")
        definition = ResolvedDefinition.model_validate_json(context.artifacts.read_bytes(command.definition))
        assert definition == checked.manifest.definition
        return construct_package(checked, context, pins=command.pins, restore_training_state=False, preflight=preflight)

    monkeypatch.setattr(inference_worker, "load_granted_package", trusted)


@pytest.mark.parametrize("name", ["standard", "looped", "nested", "custom"])
def test_actual_saved_models_generate_and_measure_in_one_resident_backend_without_optimizer_state(tmp_path, monkeypatch, name):
    """Real trained packages retain saved loops/source and deterministic evaluation across repeated chat/likelihood requests."""
    saved, _ = pretrained_policy(tmp_path, name)
    trusted_package_loader(monkeypatch, saved)
    context = InferenceContext(tmp_path / "serving")
    command = stage_load(context, saved)
    backend = InferenceBackend(context)
    try:
        assert backend.handle(command, context).outcome == "ok"
        context.pins = command.pins
        loaded = backend.loaded
        assert loaded.state.optimizer is None and not loaded.state.state_dict and not loaded.state.rng
        cfg = ChatConfig(model_name=str(uuid4()), turns=[ChatTurn(role="user", text="one")], max_tokens=8, seed=17)
        prepared = GenerationInput(config=cfg, prompt_ids=tuple(prepare_prompt(cfg, backend.tokenizer, 128)))
        grant = context.artifacts.write_bytes(encode_json(prepared), "prompt_v1")
        generate = Generate(type="generate", pins=command.pins, prompt=grant, max_tokens=8, seed=17, temperature=cfg.temperature)
        rng = torch.get_rng_state().clone()
        result = backend.handle(generate, context)
        summary = GenerationResult.model_validate_json(context.artifacts.read_bytes(result.result))
        emitted = [token for event in context.events if isinstance(event, Tokens) for token in event.token_ids]
        assert summary.text == bpe.decode(backend.tokenizer, emitted)
        assert emitted or summary.stopped == "marker"
        assert summary.generated_tokens == len(emitted)
        repeated = backend.handle(generate, context)
        again = GenerationResult.model_validate_json(context.artifacts.read_bytes(repeated.result))
        assert again.model_dump(exclude={"elapsed_secs"}) == summary.model_dump(exclude={"elapsed_secs"})
        assert backend.loaded is loaded and torch.equal(torch.get_rng_state(), rng)
        measure = MeasurementInput(config=EvalConfig(model_name=cfg.model_name, max_tokens=4, seed=17),
            text="one two three", prompts=("one", "two"))
        request = Measure(type="measure", pins=command.pins, measurement=context.artifacts.write_bytes(encode_json(measure), "measurement_v1"))
        measured = backend.handle(request, context)
        facts = MeasurementResult.model_validate_json(context.artifacts.read_bytes(measured.result))
        expected = likelihood.token_logprobs(loaded.model, bpe.encode(backend.tokenizer, measure.text))
        assert facts.likelihood.total == pytest.approx(expected.total, abs=2e-6)
        assert facts.likelihood.count == expected.count and len(facts.samples) == 2
        assert facts.model.context == 128 and facts.model.vocab == 320
        assert facts.model.parameters == sum(value.numel() for value in loaded.model.parameters())
        assert facts.model.layers is None and facts.ablation is None
        discovery = InspectionInput(model_name=cfg.model_name)
        request = Measure(type="measure", pins=command.pins, measurement=context.artifacts.write_bytes(encode_json(discovery), "measurement_v1"))
        catalog = InspectionCatalog.model_validate_json(context.artifacts.read_bytes(backend.handle(request, context).result))
        assert catalog.runtime_id == context.bootstrap.worker_id and catalog.context_window == 128
        assert all(point.evidence_hash is None for point in catalog.points)
        if name == "custom":
            assert not catalog.points and catalog.unavailable_reason
        else:
            point = catalog.points[-1]
            inspection = AttentionInput(model_name=cfg.model_name, selection=AttentionSelection(text="one", point_id=point.id, head=1))
            command_attention = Measure(type="measure", pins=command.pins, measurement=context.artifacts.write_bytes(encode_json(inspection), "measurement_v1"))
            matrix = AttentionMatrix.model_validate_json(context.artifacts.read_bytes(backend.handle(command_attention, context).result))
            assert matrix.point_id == point.id and matrix.head == 1 and matrix.runtime_id == catalog.runtime_id
            assert matrix.definition_id == command.pins.definition_id and matrix.validated_sequence_max >= len(matrix.tokens)
            current = InspectionCatalog.model_validate_json(context.artifacts.read_bytes(backend.handle(request, context).result))
            assert next(item for item in current.points if item.id == point.id).evidence_hash == matrix.evidence_hash
        assert torch.equal(torch.get_rng_state(), rng)
        context.cancelled = True
        with pytest.raises(WorkerCancelled):
            backend.handle(generate, context)
        assert backend.loaded is loaded and not any(module.training for module in loaded.model.modules())
    finally:
        backend.finish_request()
        backend.close()
    assert backend.loaded is None
    with pytest.raises(ValueError, match="closed"):
        loaded.model


def test_serving_admission_refuses_before_real_weights_are_constructed_or_read(tmp_path, monkeypatch):
    """An otherwise valid tiny saved model cannot cross a deliberately insufficient admitted host ceiling."""
    saved, _ = pretrained_policy(tmp_path, "standard")
    trusted_package_loader(monkeypatch, saved)
    context = InferenceContext(tmp_path / "small")
    resources = context.bootstrap.policy.resources.model_copy(update={"tree_resident_bytes": 1 << 20})
    context.bootstrap.policy = context.bootstrap.policy.model_copy(update={"resources": resources})
    command = stage_load(context, saved)

    def forbidden(*args, **kwargs):
        """Detect tensor deserialization after meta preflight has already proved the request cannot fit."""
        pytest.fail("insufficient serving admission reached tensor deserialization")

    monkeypatch.setattr("llm.model.checkpoint.load_state", forbidden)
    backend = InferenceBackend(context)
    with pytest.raises(ValueError, match="memory"):
        backend.handle(command, context)
    assert backend.loaded is None



def test_unsupported_attention_returns_a_static_refusal_without_custom_exception_text():
    """A supported-code validation error cannot leak administrator source details to ordinary attention callers."""
    from llm.architecture.errors import ArchitectureError
    from llm.model.inference_records import InspectionRefusal

    class RefusingInspection:
        """Expose a deliberate classified source failure for the ordinary-user response boundary."""

        def matrix(self, *args):
            """Supply forbidden diagnostic text that must remain within the numerical implementation."""
            raise ArchitectureError("inspection_budget", "PRIVATE SOURCE DIAGNOSTIC")

    backend = InferenceBackend(None)
    backend.inspection = RefusingInspection()
    prepared = AttentionInput(model_name="fixture", selection=AttentionSelection(text="a", point_id="a" * 64))
    refused = backend.attention_measurement(prepared)
    assert isinstance(refused, InspectionRefusal) and refused.code == "budget"
    assert "fewer tokens" in refused.message and "PRIVATE" not in refused.model_dump_json()
