"""Real generator recovery after checkpoint device mapping, including runnable user-deferred CUDA evidence."""

from io import BytesIO
import random

import numpy as np
import pytest
import torch

from llm.model import checkpoint


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_saved_global_generators_restore_after_checkpoint_device_mapping(device):
    """Numeric checkpoint RNG tensors can map with weights to CPU/CUDA and still restore each owning generator exactly."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("user-deferred CUDA hardware verification")
    before = checkpoint.rng_state()
    try:
        torch.manual_seed(971)
        np.random.seed(971)
        random.seed(971)
        saved = checkpoint.rng_state()
        expected = (torch.rand(8), np.random.random(8), [random.random() for _ in range(8)])
        expected_cuda = torch.rand(8, device="cuda") if device == "cuda" else None
        archive = BytesIO()
        torch.save(saved, archive)
        archive.seek(0)
        mapped = torch.load(archive, map_location=device, weights_only=True)
        assert mapped["torch"].device.type == device and mapped["numpy_keys"].device.type == device
        checkpoint.restore_rng(mapped)
        torch.testing.assert_close(torch.rand(8), expected[0], rtol=0, atol=0)
        np.testing.assert_array_equal(np.random.random(8), expected[1])
        assert [random.random() for _ in range(8)] == expected[2]
        if expected_cuda is not None:
            torch.testing.assert_close(torch.rand(8, device="cuda"), expected_cuda, rtol=0, atol=0)
    finally:
        checkpoint.restore_rng(before)
