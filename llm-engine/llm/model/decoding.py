"""Request-owned legacy caching and declared full-prefix language-model decoding."""

import torch
from torch import Tensor, nn

from llm.model.language import logits, model_facts
from llm.model.transformer import Transformer


class PrefixDecoder:
    """Own bounded ragged positions/masks and request-local cache state for generation or rollout."""

    def __init__(self, model: nn.Module, tokens: Tensor, positions: Tensor, live: Tensor, *, window: int, rope=None):
        """Evaluate a nonempty prefix under caller-owned eval/no-grad admission and explicit context bounds."""
        if torch.is_grad_enabled() or any(module.training for module in model.modules()):
            raise ValueError("decoding requires evaluation behavior and no_grad")
        facts = model_facts(model)
        if (tokens.ndim != 2 or not all(tokens.shape) or tokens.dtype != torch.int64
                or positions.shape != tokens.shape or positions.dtype != torch.int64
                or live.shape != tokens.shape or live.dtype != torch.bool
                or positions.device != tokens.device or live.device != tokens.device):
            raise ValueError("decoding requires aligned int64 tokens/positions and boolean visible keys")
        if (window < 1 or tokens.shape[1] > window or window > facts.context_window and rope is None
                or bool((positions < 0).any()) or bool((positions >= window).any())):
            raise ValueError("decoding exceeds its inherited context budget")
        if bool((tokens < 0).any()) or bool((tokens >= facts.vocabulary_size).any()):
            raise ValueError("decoding token is outside the saved vocabulary")
        if rope is not None and not isinstance(model, Transformer):
            raise ValueError("legacy positional tensors cannot configure declared decoding")
        self.model, self.tokens, self.positions, self.live = model, tokens, positions, live
        self.window, self.rope, self.closed = window, rope, False
        self.cache = model.empty_caches() if isinstance(model, Transformer) else None
        self.values = self.forward(tokens, positions)

    def forward(self, tokens: Tensor, positions: Tensor) -> Tensor:
        """Compute the next actual logits through a deliberate legacy cache adapter or the declared full-prefix contract."""
        if isinstance(self.model, Transformer):
            values, self.cache = self.model(tokens, caches=self.cache, positions=positions, key_mask=self.live, rope=self.rope)
            return values
        return logits(self.model, self.tokens, positions=self.positions, key_mask=self.live)

    def advance(self, drawn: Tensor) -> None:
        """Append one sampled token per row, preserving logical positions and visible keys before another bounded decode."""
        if self.closed or self.tokens.shape[1] >= self.window:
            raise ValueError("decoder is closed or at its context bound")
        if drawn.shape != (self.tokens.shape[0],) or drawn.dtype != torch.int64 or drawn.device != self.tokens.device:
            raise ValueError("decoder requires one int64 token per row on the admitted device")
        if bool((drawn < 0).any()) or bool((drawn >= model_facts(self.model).vocabulary_size).any()):
            raise ValueError("decoding token is outside the saved vocabulary")
        position = self.positions[:, -1:] + 1
        self.tokens = torch.cat((self.tokens, drawn[:, None]), dim=1)
        self.positions = torch.cat((self.positions, position), dim=1)
        self.live = torch.cat((self.live, torch.ones_like(position, dtype=torch.bool)), dim=1)
        self.values = self.forward(drawn[:, None], position)

    def __enter__(self):
        """Expose this request's only cache owner until completion, failure or cancellation."""
        if self.closed:
            raise ValueError("decoder is closed")
        return self

    def __exit__(self, *args) -> None:
        """Drop request tensor/model references before the next request or worker cleanup."""
        self.cache = None
        self.tokens = self.positions = self.live = self.values = self.model = None
        self.closed = True
