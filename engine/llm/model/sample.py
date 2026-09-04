"""Picking the next token from a logit vector.

Temperature, top-k and top-p compose, and each is disabled by its own neutral
value — temperature 0.0 collapses to argmax, a top_k past the vocabulary keeps
everything, a top_p of 1.0 cuts nothing. Every parameter is self-clamping, so no
combination can ask for an empty candidate set or divide by zero.
"""

import math

import torch


def pick(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token id from a 1-D `logits` tensor."""
    if logits.numel() == 0:
        return 0

    # Non-finite as well as non-positive: dividing by NaN or infinity makes every
    # weight NaN or a uniform 1.0, neither of which is a distribution. Both reach
    # here only through a bug, and argmax is the one answer always defensible.
    if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0.0:
        return int(logits.argmax().item())

    logits = logits.float() / temperature

    if 0 < top_k < logits.numel():
        kept, _ = torch.topk(logits, top_k)
        logits = logits.masked_fill(logits < kept[-1], float("-inf"))

    if 0.0 < top_p < 1.0:
        ordered, order = torch.sort(logits, descending=True)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        # Shift so the token that crosses the threshold is itself kept — without
        # it a top_p below the top token's own probability keeps nothing.
        drop = cumulative - torch.softmax(ordered, dim=-1) >= top_p
        ordered = ordered.masked_fill(drop, float("-inf"))
        logits = torch.empty_like(logits).scatter_(0, order, ordered)

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def pick_rows(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token id per row of a 2-D `logits` [batch, vocab].

    The row-wise twin of `pick`, with the same self-clamping neutral values and
    the same order of operations, for a caller generating a batch of sequences in
    lockstep. It is a separate function rather than a generalization because
    `pick` is on the serving path and returns an `int` that a dozen callers
    index and compare with — widening it to a tensor would move that cost onto
    every one of them for a batch of one.
    """
    batch, vocab = logits.shape
    if vocab == 0:
        return torch.zeros(batch, dtype=torch.long, device=logits.device)

    if not math.isfinite(temperature) or temperature <= 0.0:
        return logits.argmax(dim=-1)

    logits = logits.float() / temperature

    if 0 < top_k < vocab:
        kept, _ = torch.topk(logits, top_k, dim=-1)
        logits = logits.masked_fill(logits < kept[:, -1:], float("-inf"))

    if 0.0 < top_p < 1.0:
        ordered, order = torch.sort(logits, dim=-1, descending=True)
        probs = torch.softmax(ordered, dim=-1)
        # Shift so the token that crosses the threshold is itself kept — without
        # it a top_p below the top token's own probability keeps nothing.
        drop = probs.cumsum(dim=-1) - probs >= top_p
        ordered = ordered.masked_fill(drop, float("-inf"))
        logits = torch.empty_like(logits).scatter_(1, order, ordered)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
