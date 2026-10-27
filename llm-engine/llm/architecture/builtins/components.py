"""Readable version-one named-port blocks retaining Slate Transformer arithmetic.

The legacy tensor primitives remain the numerical authority. These adapters do
not enable legacy diagnostic mutation or keep request state on model instances.
"""

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from llm.architecture.errors import ArchitectureError
from llm.architecture.interfaces import ExecutionOccurrence, InspectionPoint
from llm.architecture.sdk import block, BlockDeclaration, ParameterDeclaration, TensorPort
from llm.model.transformer import Attention, Block, FeedForward, rope_tables


def integer(default: int, minimum: int = 1) -> ParameterDeclaration:
    """Declare one positive bounded-by-worker integer constructor control."""
    return ParameterDeclaration(type="integer", default=default, minimum=minimum)


def hidden(width: str = "width") -> TensorPort:
    """Describe a floating batch of sequence-local hidden vectors."""
    return TensorPort(dtype="floating", shape=("batch", "sequence", width))


TOKEN_PORT = TensorPort(dtype="int64", shape=("batch", "sequence"))
MASK_PORT = TensorPort(dtype="bool", shape=("batch", "sequence"))
LOGIT_PORT = TensorPort(dtype="floating", shape=("batch", "sequence", "vocabulary"))
LAYER_PARAMETERS = {
    "dimensions": integer(32, 2), "heads": integer(4), "ff_multiplier": integer(4),
    "ctx_window": integer(32),
    "rope_base": ParameterDeclaration(type="number", default=10000.0, minimum=2.0),
    "learned_temperature": ParameterDeclaration(type="boolean", default=False),
}


def check_attention(dimensions: int, heads: int, ctx_window: int, rope_base: float) -> None:
    """Reject invalid head/RoPE dimensions before allocating builtin parameters or tables."""
    if dimensions < 2 or heads < 1 or dimensions % heads or (dimensions // heads) % 2:
        raise ArchitectureError("attention_dimensions", "dimensions must contain an even positive size per head")
    if ctx_window < 1 or not math.isfinite(rope_base) or rope_base < 2:
        raise ArchitectureError("attention_positions", "context must be positive and RoPE base finite and at least two")


def initialize(module: nn.Module) -> None:
    """Apply Slate's Xavier projection/embedding initialization and zero linear biases."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)


@block(BlockDeclaration(sdk_version=1, name="token_embedding", title="Token embedding",
    parameters={"vocab": integer(32), "width": integer(32)},
    inputs={"tokens": TOKEN_PORT}, outputs={"hidden": hidden()}))
class TokenEmbedding(nn.Module):
    """Tokenizer-bound lookup table with an exact aliasable vocabulary-by-width Parameter."""

    def __init__(self, vocab: int, width: int):
        """Allocate and Xavier-initialize the table from explicit vocabulary/width values."""
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab, width))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, tokens: Tensor) -> dict[str, Tensor]:
        """Gather hidden vectors without changing input IDs or table state."""
        return {"hidden": F.embedding(tokens, self.weight)}


@block(BlockDeclaration(sdk_version=1, name="normalization", title="Layer normalization",
    parameters={"width": integer(32), "epsilon": ParameterDeclaration(type="number", default=1e-5, minimum=1e-12)},
    inputs={"hidden": hidden()}, outputs={"hidden": hidden()}))
class Normalization(nn.Module):
    """Affine LayerNorm with the legacy biased variance, scale and bias policy."""

    def __init__(self, width: int, epsilon: float):
        """Create unit scale and zero bias for the configured last dimension and epsilon."""
        super().__init__()
        self.norm = nn.LayerNorm(width, eps=epsilon)

    def forward(self, hidden: Tensor) -> dict[str, Tensor]:
        """Normalize each token's channels independently, retaining gradients."""
        return {"hidden": self.norm(hidden)}


@block(BlockDeclaration(sdk_version=1, name="feed_forward", title="GELU feed forward",
    parameters={"dimensions": integer(32), "ff_multiplier": integer(4)},
    inputs={"hidden": hidden("dimensions")}, outputs={"hidden": hidden("dimensions")}))
class FeedForwardBlock(nn.Module):
    """Two biased projections with exact GELU, matching the existing feed-forward block."""

    def __init__(self, dimensions: int, ff_multiplier: int):
        """Construct and initialize one independently owned feed-forward component."""
        super().__init__()
        self.ff = FeedForward(dimensions, ff_multiplier)
        self.apply(initialize)

    def forward(self, hidden: Tensor) -> dict[str, Tensor]:
        """Project each hidden token through the configured GELU bottleneck."""
        return {"hidden": self.ff(hidden)}


@block(BlockDeclaration(sdk_version=1, name="add", title="Residual addition", parameters={},
    inputs={"left": hidden(), "right": hidden()}, outputs={"hidden": hidden()}))
class Add(nn.Module):
    """An explicit merge with equal-shaped named branch inputs."""

    def __init__(self):
        """Construct a stateless merge with no trainable parameters."""
        super().__init__()

    def forward(self, left: Tensor, right: Tensor) -> dict[str, Tensor]:
        """Add equal-shaped tensors, rejecting implicit broadcast merges."""
        if left.shape != right.shape:
            raise ArchitectureError("merge_shape", "addition requires identical branch shapes")
        return {"hidden": left + right}


@block(BlockDeclaration(sdk_version=1, name="concat", title="Channel concatenation", parameters={},
    inputs={"left": hidden("left_width"), "right": hidden("right_width")},
    outputs={"hidden": hidden("output_width")}))
class Concat(nn.Module):
    """Concatenate channels explicitly while preserving batch and sequence axes."""

    def __init__(self):
        """Construct a stateless channel merge."""
        super().__init__()

    def forward(self, left: Tensor, right: Tensor) -> dict[str, Tensor]:
        """Return joined channels, rejecting mismatched leading dimensions."""
        if left.shape[:-1] != right.shape[:-1]:
            raise ArchitectureError("merge_shape", "concatenation requires matching batch and sequence axes")
        return {"hidden": torch.cat((left, right), dim=-1)}


@block(BlockDeclaration(sdk_version=1, name="language_model_head", title="Vocabulary projection",
    parameters={"vocab": integer(32), "width": integer(32)},
    inputs={"hidden": hidden()}, outputs={"logits": LOGIT_PORT}))
class OutputProjection(nn.Module):
    """Bias-free output projection supporting exact Parameter identity with token embedding."""

    def __init__(self, vocab: int, width: int):
        """Allocate a vocabulary-by-width matrix; recipes explicitly tie its owner when needed."""
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab, width))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, hidden: Tensor) -> dict[str, Tensor]:
        """Return vocabulary logits by multiplying the transpose of the owned/shared matrix."""
        return {"logits": hidden @ self.weight.T}


class PositionedBlock(nn.Module):
    """Derived nonpersistent RoPE tables shared by attention and whole-layer adapters."""

    def __init__(self, dimensions: int, heads: int, ctx_window: int, rope_base: float):
        """Validate attention dimensions and reconstruct deterministic position buffers."""
        super().__init__()
        check_attention(dimensions, heads, ctx_window, rope_base)
        cos, sin = rope_tables(dimensions // heads, ctx_window, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)


@block(BlockDeclaration(sdk_version=1, name="causal_attention", title="Causal RoPE attention",
    parameters={name: value for name, value in LAYER_PARAMETERS.items() if name != "ff_multiplier"},
    inputs={"hidden": hidden("dimensions"), "positions": TOKEN_PORT, "key_mask": MASK_PORT},
    outputs={"hidden": hidden("dimensions")}, capabilities=("introspection", "attention")))
class CausalAttention(PositionedBlock):
    """Fused QKV attention with adjacent-pair RoPE and optional learned per-head temperature."""

    def __init__(self, dimensions: int, heads: int, ctx_window: int, rope_base: float, learned_temperature: bool):
        """Create one attention instance with legacy initialization and no retained request cache."""
        super().__init__(dimensions, heads, ctx_window, rope_base)
        self.attn = Attention(dimensions, heads, learned_temperature)
        self.apply(initialize)
        self.attn.init_projections()

    def forward(self, hidden: Tensor, positions: Tensor, key_mask: Tensor) -> dict[str, Tensor]:
        """Attend causally using explicit logical positions and visible keys."""
        result, _ = self.attn(hidden, self.rope_cos[positions], self.rope_sin[positions], key_mask=key_mask)
        return {"hidden": result}

    def inspection_points(self) -> tuple[InspectionPoint, ...]:
        """Declare the actual local attention view; composition and runtime evidence own occurrence identity and enablement."""
        return (InspectionPoint("attention", ExecutionOccurrence("", ()), "attention", "attention", "Attention", self.attn.heads),)

    def inspect_named(self, inputs: Mapping[str, Tensor], *, point_id: str) -> Tensor:
        """Measure canonical attention on supplied hidden values and explicit positions/mask, retaining no diagnostic state."""
        if point_id != "attention":
            raise ArchitectureError("inspection_point", "unknown causal-attention inspection point")
        positions = inputs["positions"]
        return self.attn.attention_weights(inputs["hidden"], self.rope_cos[positions], self.rope_sin[positions], inputs["key_mask"])


@block(BlockDeclaration(sdk_version=1, name="transformer_layer", title="Pre-LN Transformer layer",
    parameters=LAYER_PARAMETERS,
    inputs={"hidden": hidden("dimensions"), "positions": TOKEN_PORT, "key_mask": MASK_PORT},
    outputs={"hidden": hidden("dimensions")}, capabilities=("introspection", "attention")))
class TransformerLayer(PositionedBlock):
    """Reusable attention and GELU residual composite preserving legacy layer arithmetic."""

    def __init__(self, dimensions: int, heads: int, ff_multiplier: int, ctx_window: int,
                 rope_base: float, learned_temperature: bool):
        """Create an independent pre-normalized layer and reconstruct its position tables."""
        super().__init__(dimensions, heads, ctx_window, rope_base)
        self.layer = Block(dimensions, heads, ff_multiplier, learned_temperature)
        self.apply(initialize)
        self.layer.attn.init_projections()

    def forward(self, hidden: Tensor, positions: Tensor, key_mask: Tensor) -> dict[str, Tensor]:
        """Return the residual stream after one fixed, side-effect-free layer application."""
        result, _ = self.layer(hidden, self.rope_cos[positions], self.rope_sin[positions], key_mask=key_mask)
        return {"hidden": result}

    def inspection_points(self) -> tuple[InspectionPoint, ...]:
        """Declare this layer's real attention view without multiplying its shared parameter ownership."""
        return (InspectionPoint("attention", ExecutionOccurrence("", ()), "attention", "attention", "Attention", self.layer.attn.heads),)

    def inspect_named(self, inputs: Mapping[str, Tensor], *, point_id: str) -> Tensor:
        """Measure attention after the same pre-normalization used by the layer's normal forward, without recording flags."""
        if point_id != "attention":
            raise ArchitectureError("inspection_point", "unknown Transformer-layer inspection point")
        positions = inputs["positions"]
        return self.layer.attn.attention_weights(self.layer.ln_attn(inputs["hidden"]),
            self.rope_cos[positions], self.rope_sin[positions], inputs["key_mask"])
