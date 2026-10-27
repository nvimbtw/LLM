"""Version-one standard composite with deliberately unchanged legacy construction order."""

import torch
from torch import Tensor, nn

from llm.architecture.builtins.components import (
    LAYER_PARAMETERS, LOGIT_PORT, MASK_PORT, TOKEN_PORT, check_attention, integer,
)
from llm.architecture.errors import ArchitectureError
from llm.architecture.interfaces import ExecutionOccurrence, InspectionPoint, TokenRowResult
from llm.architecture.sdk import block, BlockDeclaration
from llm.model.transformer import Transformer


@block(BlockDeclaration(sdk_version=1, name="standard_transformer", title="Standard Transformer",
    parameters={"vocab": integer(32), **LAYER_PARAMETERS, "layers": integer(2)},
    inputs={"tokens": TOKEN_PORT, "positions": TOKEN_PORT, "key_mask": MASK_PORT},
    outputs={"logits": LOGIT_PORT}, capabilities=("token_rows", "recompute", "introspection", "attention")))
class StandardTransformer(nn.Module):
    """Named-port standard model preserving tied weights, RNG order and all legacy math.

    The pinned legacy primitive is part of the builtin compatibility manifest.
    Optional methods belong to this block; a composed model must explicitly
    adapt and validate them before offering a whole-model capability.
    """

    def __init__(self, vocab: int, dimensions: int, heads: int, ff_multiplier: int,
                 ctx_window: int, rope_base: float, learned_temperature: bool, layers: int):
        """Construct the exact legacy Transformer once, retaining its initial RNG consumption."""
        super().__init__()
        check_attention(dimensions, heads, ctx_window, rope_base)
        self.model = Transformer(vocab=vocab, dimensions=dimensions, heads=heads, layers=layers,
                                 ff_multiplier=ff_multiplier, ctx_window=ctx_window,
                                 rope_base=rope_base, learned_temperature=learned_temperature)

    def forward(self, tokens: Tensor, positions: Tensor, key_mask: Tensor) -> dict[str, Tensor]:
        """Return named full-prefix logits with explicit padding and logical positions."""
        logits, _ = self.model(tokens, positions=positions, key_mask=key_mask)
        return {"logits": logits}

    def initialize_token_rows(self, token_ids: tuple[int, ...]) -> TokenRowResult:
        """Initialize unique in-range tied rows to the original mean, reporting skipped IDs."""
        return self.model.initialize_token_rows(token_ids)

    def inspection_points(self) -> tuple[InspectionPoint, ...]:
        """Expose actual independent standard layers under local module/call identities, with known head counts."""
        return tuple(InspectionPoint(f"layer-{index}", ExecutionOccurrence(f"blocks/{index}", (index,)),
            "attention", "attention", f"Layer {index + 1} attention", self.model.heads) for index in range(self.model.layers))

    def inspect(self, tokens: Tensor, *, point_id: str, positions: Tensor | None = None,
                key_mask: Tensor | None = None) -> Tensor:
        """Measure one actual saved layer after the preceding residual computations, preserving flags and request tensors."""
        selected = next((index for index in range(self.model.layers) if point_id == f"layer-{index}"), None)
        if selected is None:
            raise ArchitectureError("inspection_point", "unknown standard Transformer inspection point")
        if any(block.record_activations or block.attn.record_attention for block in self.model.blocks):
            raise ArchitectureError("inspection_state", "standard attention inspection requires inactive legacy recording")
        if positions is None:
            positions = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        if key_mask is None:
            key_mask = torch.ones_like(tokens, dtype=torch.bool)
        cos, sin = self.model.rope_cos[positions], self.model.rope_sin[positions]
        hidden = self.model.embedding(tokens)
        for index, block in enumerate(self.model.blocks):
            if index == selected:
                return block.attn.attention_weights(block.ln_attn(hidden), cos, sin, key_mask)
            hidden, _ = block(hidden, cos, sin, key_mask=key_mask)
        raise ArchitectureError("inspection_point", "standard attention layer was not executed")

    def recomputation_targets(self) -> tuple[ExecutionOccurrence, ...]:
        """Expose legal independent standard-layer recomputation identities."""
        return tuple(ExecutionOccurrence(f"blocks/{index}", (index,)) for index in range(self.model.layers))

    def configure_recomputation(self, occurrences: tuple[ExecutionOccurrence, ...]) -> None:
        """Set exact validated layer occurrences between steps without changing model identity."""
        targets = self.recomputation_targets()
        if len(set(occurrences)) != len(occurrences) or any(value not in targets for value in occurrences):
            raise ArchitectureError("recompute_occurrence", "unknown or duplicate standard layer occurrence")
        self.model._recomputed = frozenset(value.call_path[0] for value in occurrences)
        self.model._recompute_depth = len(occurrences)
