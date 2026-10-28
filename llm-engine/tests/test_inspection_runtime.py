"""Actual worker-local attention proof, memory admission, determinism and cancellation without process-containment claims."""

from uuid import uuid4

import pytest
import torch

from llm.architecture.errors import ArchitectureError
from llm.architecture.spec import canonical_digest
from llm.config import TrainConfig
from llm.orchestrate.inspection import InspectionRuntime
from llm.runtime.protocol import RuntimePins
from llm.runtime.worker import WorkerCancelled
from llm.train.model_preflight import training_profile
from tests.test_architecture_builtins import build_template
from tests.test_modular_attention import custom_attention_model
from tests.test_modular_worker import NumericalContext


def runtime(model, tmp_path):
    """Create finite numerical test grants/context and exact profile pins; this seam supplies no guardian or admission certificate."""
    context = NumericalContext(tmp_path / "inspection")
    context.bootstrap.worker_id = uuid4()
    profile = training_profile(TrainConfig(model_name="fixture", batch_size=2, precision="fp32"), model.facts.context_window, "cpu")
    pins = RuntimePins(definition_id="a" * 64, bindings_digest="b" * 64,
        profile_digest=canonical_digest(profile), receipt_id=uuid4())
    return InspectionRuntime(model, context, pins=pins, profile=profile), context


@pytest.mark.parametrize("name", ["standard", "looped", "nested"])
def test_actual_attention_requires_complete_view_evidence_and_reuses_only_proved_bounds(tmp_path, name):
    """Real selected views pass meta admission, strict singleton/range capture and numerical checks before returning an independent CPU matrix."""
    model = build_template(name).train()
    active, context = runtime(model, tmp_path)
    point = active.bindings()[-1].point
    modes = tuple(module.training for module in model.modules())
    rng = torch.get_rng_state().clone()
    measured = active.attention(torch.tensor([[1, 2, 3, 4]]), point_id=point.id, head=1)
    assert measured.weights.shape == (4, 4) and measured.weights.device.type == "cpu"
    assert measured.evidence.sequence_max == 4 and measured.evidence.heads == 2
    assert len(measured.evidence.capture_hashes) == 2 and len(measured.evidence.evidence_hash) == 64
    assert active.proven[point.id] == measured.evidence
    expected = model.inspect(torch.tensor([[1, 2, 3, 4]]), point_id=point.id)[0, 1]
    torch.testing.assert_close(measured.weights, expected, rtol=2e-5, atol=2e-6)
    shorter = active.attention(torch.tensor([[1, 2]]), point_id=point.id, head=0)
    assert shorter.evidence == measured.evidence
    larger = active.attention(torch.tensor([[1, 2, 3, 4, 5]]), point_id=point.id, head=0)
    assert larger.evidence.sequence_max == 5 and larger.evidence.evidence_hash != measured.evidence.evidence_hash
    assert torch.equal(torch.get_rng_state(), rng)
    assert tuple(module.training for module in model.modules()) == modes
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    active.bindings()
    assert not active.proven
    active.close()
    with pytest.raises(ValueError, match="closed"):
        active.bindings()


def test_actual_custom_attention_is_proved_before_use(tmp_path):
    """A real delegated custom source point passes the same complete capture/probability path as supplied recipes."""
    with custom_attention_model() as model:
        active, _ = runtime(model, tmp_path)
        point = active.bindings()[0].point
        measured = active.attention(torch.tensor([[1, 2, 3]]), point_id=point.id, head=0)
        assert measured.weights.shape == (3, 3) and measured.evidence.heads == 2
        assert len(active.proven) == 1
        active.close()


def test_inspection_budget_and_cancellation_cannot_publish_partial_capability_evidence(tmp_path):
    """Impossible snapshots refuse before metadata execution; cancellation between validation phases leaves no reusable proof."""
    model = build_template("looped")
    active, context = runtime(model, tmp_path)
    point = active.bindings()[0].point
    resources = context.bootstrap.policy.resources
    context.bootstrap.policy = context.bootstrap.policy.model_copy(update={"resources": resources.model_copy(update={"tree_resident_bytes": 1 << 20})})
    with pytest.raises(ArchitectureError, match="snapshots"):
        active.attention(torch.tensor([[1, 2]]), point_id=point.id, head=0)
    assert not active.proven
    context.bootstrap.policy = context.bootstrap.policy.model_copy(update={"resources": resources})
    context.cancelled = True
    with pytest.raises(WorkerCancelled):
        active.attention(torch.tensor([[1, 2]]), point_id=point.id, head=0)
    assert not active.proven
    active.close()


def test_missing_points_and_invalid_heads_or_prefixes_do_not_enable_measurement(tmp_path):
    """Selection and resource errors cannot authorize another occurrence or a substituted Transformer layout."""
    model = build_template("standard")
    active, _ = runtime(model, tmp_path)
    point = active.bindings()[0].point
    for tokens, identity, head in ((torch.tensor([[1, 2]]), "missing", 0),
            (torch.tensor([[1, 2]]), point.id, 2), (torch.tensor([[1.0, 2.0]]), point.id, 0),
            (torch.tensor([[1] * 13]), point.id, 0)):
        with pytest.raises(ArchitectureError):
            active.attention(tokens, point_id=identity, head=head)
    assert not active.proven
    active.close()
