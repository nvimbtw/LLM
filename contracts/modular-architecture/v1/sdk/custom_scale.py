"""Version-one source contract example; the SDK implementation belongs to ISS-50."""

import torch
from torch import Tensor, nn

from llm.architecture.sdk import (
    BlockDeclaration, ParameterDeclaration, TensorPort, block,
)


@block(BlockDeclaration(
    sdk_version=1,
    name="custom_scale",
    title="Custom Scale",
    description="A learned, token-local residual scale with named tensor ports.",
    parameters={"gain": ParameterDeclaration(
        type="number", default=0.25, minimum=-2.0, maximum=2.0,
        title="Gain", description="Initial value of the learned scale.",
    )},
    inputs={"hidden": TensorPort(dtype="floating", shape=("batch", "sequence", "width"))},
    outputs={"hidden": TensorPort(dtype="floating", shape=("batch", "sequence", "width"))},
))
class CustomScale(nn.Module):
    """Multiply each token's hidden channels by one learned scalar."""

    def __init__(self, gain: float):
        """Initialize the one parameter from its validated constructor value."""
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(gain, dtype=torch.float32))

    def forward(self, hidden: Tensor) -> dict[str, Tensor]:
        """Return named token-local output without mutating input or persistent state."""
        return {"hidden": hidden * self.gain}
