"""Runnable deferred CUDA attention evidence; numerical seams do not certify OS worker containment."""

from uuid import uuid4

import pytest
import torch

from llm.architecture.spec import canonical_digest
from llm.config import TrainConfig
from llm.orchestrate.inspection import InspectionRuntime
from llm.runtime.protocol import RuntimePins
from llm.train.model_preflight import training_profile
from tests.test_architecture_builtins import build_template
from tests.test_modular_worker import NumericalContext


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware explicitly deferred by user")
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("name", ["standard", "looped", "nested"])
def test_cuda_occurrence_attention_preserves_rng_and_proves_exact_precision_domain(tmp_path, name, precision):
    """Capture/measure actual shared/nested CUDA views under their precision profile while retaining request-independent CPU output."""
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("selected CUDA device has no bf16 support")
    model = build_template(name).to("cuda").eval()
    context = NumericalContext(tmp_path / "cuda-inspection")
    context.device = "cuda"
    context.bootstrap.worker_id = uuid4()
    resources = context.bootstrap.policy.resources.model_copy(update={"device": "cuda", "device_allocator_bytes": 256 << 20})
    context.bootstrap.policy = context.bootstrap.policy.model_copy(update={"resources": resources})
    profile = training_profile(TrainConfig(model_name="fixture", batch_size=2, precision=precision), model.facts.context_window, "cuda")
    pins = RuntimePins(definition_id="a" * 64, bindings_digest="b" * 64,
        profile_digest=canonical_digest(profile), receipt_id=uuid4())
    active = InspectionRuntime(model, context, pins=pins, profile=profile)
    rng = torch.cuda.get_rng_state().clone()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            point = active.catalog("fixture").points[-1]
            measured = active.attention(torch.tensor([[1, 2, 3, 4]], device="cuda"), point_id=point.id, head=1)
        assert measured.weights.device.type == "cpu" and measured.weights.dtype == torch.float32
        assert measured.evidence.profile_digest == pins.profile_digest and measured.evidence.sequence_max == 4
        assert torch.equal(torch.cuda.get_rng_state(), rng)
    finally:
        active.close()
