"""Teacher-forced model likelihood, without benchmark scoring policy."""
import math
from dataclasses import dataclass
import torch
from llm.model.transformer import Transformer

@dataclass
class TokenScores:
    """Per-token natural-log probabilities under teacher forcing."""

    logprobs: list[float]

    @property
    def total(self) -> float:
        return sum(self.logprobs)

    @property
    def count(self) -> int:
        return len(self.logprobs)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.logprobs else math.nan


@torch.inference_mode()
def token_logprobs(model: Transformer, tokens: list[int], device: str = "cpu", cancel=None) -> TokenScores:
    """Natural-log probability of each token given everything before it.

    The first token is context, not a prediction, so it is never scored.

    Sequences longer than the context window are scored in strided chunks that
    each carry half a window of context, so no token is predicted from a cold
    start beyond the very first — scoring disjoint windows would charge the model
    for tokens it was given no context for and understate it badly.
    """
    if len(tokens) < 2:
        return TokenScores([])

    window = model.ctx_window
    stride = max(1, window // 2)
    logprobs: list[float] = []
    scored_upto = 1  # index of the next token needing a score

    start = 0
    while scored_upto < len(tokens):
        if cancel is not None and cancel.is_set():
            break
        chunk = tokens[start : start + window]
        if len(chunk) < 2:
            break

        logits, _ = model(torch.tensor([chunk], device=device))
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)

        # Position i predicts token i+1.
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


