"""Worker math-policy lifecycle controls only; backend-flag tests do not claim CUDA allocation or numerical hardware evidence."""

import pytest
import torch

from llm.architecture.worker import validation_math_settings
from tests.test_architecture_validation import profile, recipe_snapshot
from llm.architecture.validation import validate_snapshot


@pytest.mark.parametrize("cancel", [False, True])
def test_requested_tf32_settings_are_scoped_and_restored_without_gpu_execution(cancel):
    """Actual backend flags follow the requested policy and restore through both ordinary exit and cancellation, without running CUDA tensors."""
    before = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if cancel:
            with pytest.raises(InterruptedError):
                with validation_math_settings(profile(allow_tf32=True), "cuda"):
                    assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
                    raise InterruptedError("request cancelled")
        else:
            with validation_math_settings(profile(allow_tf32=True), "cuda"):
                assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before


def test_inspection_and_cpu_requests_do_not_change_backend_math_flags():
    """Requests without CUDA numerical work cannot silently rewrite process-wide CUDA policy."""
    before = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    for declaration, device in ((None, "cuda"), (profile(allow_tf32=True), "cpu")):
        with validation_math_settings(declaration, device):
            assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA validation hardware is user-deferred on this CPU machine")
@pytest.mark.parametrize("allow_tf32", [False, True])
def test_cuda_validation_captures_under_requested_tf32_policy(allow_tf32):
    """Run actual CUDA strict captures/numerical preflight with explicit TF32 and restore the caller's settings; remains runnable on the user's GPU machine."""
    before = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    snapshot = recipe_snapshot(profile=profile(device="cuda", allow_tf32=allow_tf32))
    with validation_math_settings(snapshot.request.profile, "cuda"):
        output = validate_snapshot(snapshot, device="cuda")
        assert output.report.verdict == "valid", output.report.diagnostics
        assert output.report.receipt.identity.profile.allow_tf32 is allow_tf32
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before
