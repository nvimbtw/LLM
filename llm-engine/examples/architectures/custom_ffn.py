"""Copyable custom gated FFN; use a new immutable block revision when editing."""

import torch
from torch import nn
from llm.architecture.sdk import block, BlockDeclaration, ParameterDeclaration, TensorPort


@block(BlockDeclaration(sdk_version=1, name="custom_gated_ffn", title="Custom gated FFN",
    parameters={"width": ParameterDeclaration(type="integer", default=32, minimum=1),
                "expansion": ParameterDeclaration(type="integer", default=3, minimum=1)},
    inputs={"hidden": TensorPort(dtype="floating", shape=("batch", "sequence", "width"))},
    outputs={"hidden": TensorPort(dtype="floating", shape=("batch", "sequence", "width"))}))
class CustomGatedFFN(nn.Module):
    """A genuinely custom SiLU gated residual replacing the standard GELU FFN."""

    def __init__(self, width: int, expansion: int):
        """Allocate independent gate/value/up projections from exposed typed controls."""
        super().__init__()
        self.gate = nn.Linear(width, width * expansion)
        self.value = nn.Linear(width, width * expansion)
        self.output = nn.Linear(width * expansion, width)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a token-local gated residual with ordinary PyTorch autograd."""
        gated = torch.nn.functional.silu(self.gate(hidden)) * self.value(hidden)
        return {"hidden": hidden + self.output(gated)}
