"""Contract checks for a language model without Transformer-specific attributes."""

from collections.abc import Mapping
from uuid import UUID

import torch
from torch import nn

from llm.architecture.interfaces import (
    BlockCachedResult, CacheIdentity, CachedBlock, CachedLanguageModel, ContextExtensibleModel,
    ExecutionOccurrence, LanguageModel, LanguageModelFacts, ModelIntrospection,
    RecomputableModel, RequestCache, TokenRowInitializer,
)


class TokenLocalModel(nn.Module):
    """A causal token-local model exposing no embedding or blocks convention."""

    def __init__(self):
        """Create a small vocabulary projection owned by ordinary PyTorch."""
        super().__init__()
        self.projection = nn.Linear(9, 9)

    @property
    def facts(self) -> LanguageModelFacts:
        """Expose final vocabulary/context facts without executing the model."""
        return LanguageModelFacts(vocabulary_size=9, context_window=16)

    def forward(
        self, tokens: torch.Tensor, *, positions: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project tokens independently; padding inputs cannot affect live outputs."""
        return self.projection(torch.nn.functional.one_hot(tokens, 9).float())


def test_full_prefix_model_does_not_require_optional_capabilities():
    """Finite next-token gradients need neither optional capabilities nor fixed layout fields."""
    model = TokenLocalModel().eval()
    assert isinstance(model, LanguageModel)
    assert not hasattr(model, "embedding") and not hasattr(model, "blocks")
    for optional in (CachedLanguageModel, ModelIntrospection, RecomputableModel,
                     TokenRowInitializer, ContextExtensibleModel):
        assert not isinstance(model, optional)
    tokens = torch.tensor([[1, 2, 3, 4]])
    logits = model(tokens)
    assert logits.shape == (1, 4, model.facts.vocabulary_size)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 9), tokens[:, 1:].reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_explicit_inputs_preserve_prefix_and_batch_independence():
    """A non-Transformer model accepts explicit adapter-supplied positions and masks."""
    model = TokenLocalModel().eval()
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    positions = torch.arange(3).expand(2, -1)
    mask = torch.tensor([[False, True, True], [True, True, True]])
    actual = model(tokens, positions=positions, key_mask=mask)
    torch.testing.assert_close(actual[0, 1:], model(tokens[:1, 1:])[0])
    torch.testing.assert_close(actual[:, :2], model(tokens[:, :2]))


def test_parameter_owner_does_not_collapse_loop_request_state():
    """Occurrence keys retain same-module cache tensors across nested loop positions."""
    first = ExecutionOccurrence("core", (1, 0, 0))
    second = ExecutionOccurrence("core", (1, 0, 1))
    identity = CacheIdentity(UUID("11111111-1111-4111-8111-111111111111"), "a" * 64)
    cache = RequestCache(identity, {first: {"keys": torch.ones(1, 2, 3, 4)},
                                    second: {"keys": torch.zeros(1, 2, 3, 4)}})
    assert first.instance_path == second.instance_path
    assert first != second and len(cache.entries) == 2
    assert cache.entries[first]["keys"].data_ptr() != cache.entries[second]["keys"].data_ptr()
    assert cache.identity != CacheIdentity(UUID("22222222-2222-4222-8222-222222222222"), "a" * 64)
    assert cache.identity != CacheIdentity(identity.request_id, "b" * 64)


class PrefixSumBlock(nn.Module):
    """A causal hidden-value block with explicit state and no token/logit assumptions."""

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return causal cumulative hidden values for a complete prefix."""
        return {"hidden": hidden.cumsum(dim=1)}

    def cached_forward(
        self, inputs: Mapping[str, torch.Tensor], state: Mapping[str, torch.Tensor],
    ) -> BlockCachedResult:
        """Continue cumulative values from supplied state without mutating caller tensors."""
        hidden = inputs["hidden"]
        previous = state.get("sum", torch.zeros_like(hidden[:, :1]))
        result = hidden.cumsum(dim=1) + previous
        return BlockCachedResult({"hidden": result}, {"sum": result[:, -1:].clone()})


def test_internal_cached_block_has_explicit_independent_hidden_state():
    """One hidden block serves separate occurrence states and matches full-prefix outputs."""
    block = PrefixSumBlock()
    assert isinstance(block, CachedBlock)
    assert not isinstance(block, CachedLanguageModel)
    prefix = torch.randn(2, 3, 8)
    next_hidden = torch.randn(2, 1, 8)
    initial = prefix.clone()
    prefilled = block.cached_forward({"hidden": prefix}, {})
    before = prefilled.state["sum"].clone()
    decoded = block.cached_forward({"hidden": next_hidden}, prefilled.state)
    expected = block(torch.cat((prefix, next_hidden), dim=1))["hidden"][:, -1:]
    torch.testing.assert_close(decoded.outputs["hidden"], expected)
    torch.testing.assert_close(prefilled.state["sum"], before)
    torch.testing.assert_close(prefix, initial)
    separate = block.cached_forward({"hidden": next_hidden}, {})
    torch.testing.assert_close(separate.outputs["hidden"], next_hidden)
    assert separate.state["sum"].data_ptr() != decoded.state["sum"].data_ptr()
