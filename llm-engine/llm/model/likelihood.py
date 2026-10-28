"""Teacher-forced model likelihood, without benchmark scoring policy."""
import math
from dataclasses import dataclass
import torch
from llm.model.language import evaluating, logits, model_facts

@dataclass
class TokenScores:
    """Per-token natural-log probabilities under teacher forcing."""

    logprobs: list[float]

    @property
    def total(self) -> float:
        """Return the sum of all measured natural-log probabilities."""
        return sum(self.logprobs)

    @property
    def count(self) -> int:
        """Return the number of predicted tokens, excluding the initial context token."""
        return len(self.logprobs)

    @property
    def mean(self) -> float:
        """Return mean log probability, or NaN when no prediction was measured."""
        return self.total / self.count if self.logprobs else math.nan


@torch.no_grad()
def token_logprobs(model: torch.nn.Module, tokens: list[int], device: str = "cpu", cancel=None) -> TokenScores:
    """Natural-log probability of each token given everything before it.

    The first token is context, not a prediction, so it is never scored.

    Sequences longer than the context window are scored in strided chunks that
    each carry half a window of context, so no token is predicted from a cold
    start beyond the very first — scoring disjoint windows would charge the model
    for tokens it was given no context for and understate it badly.
    """
    if len(tokens) < 2:
        return TokenScores([])

    window = model_facts(model).context_window
    if window < 2:
        raise ValueError("likelihood requires at least two tokens of model context")
    stride = max(1, window // 2)
    logprobs: list[float] = []
    scored_upto = 1

    start = 0
    while scored_upto < len(tokens):
        if cancel is not None and cancel.is_set():
            break
        chunk = tokens[start : start + window]
        if len(chunk) < 2:
            break

        with evaluating(model):
            values = logits(model, torch.tensor([chunk], device=device))
        log_probs = torch.log_softmax(values[0].float(), dim=-1)

        for i in range(len(chunk) - 1):
            index = start + i + 1
            if index < scored_upto:
                continue
            logprobs.append(float(log_probs[i, chunk[i + 1]]))
            scored_upto = index + 1

        if start + window >= len(tokens):
            break
        start += stride

    return TokenScores(logprobs)


