"""Token log-probabilities, and the composite score built on them."""

import math
from dataclasses import dataclass

import torch

from llm.model.transformer import Transformer

SCORE_VERSION = 2

# Held-out bits per byte, through a logistic: strictly monotone, smooth, and
# asymptotic at both ends, so it never clamps and improvement always pays.
#
# MID sits at the centre of the range these models occupy, which is where the
# response is steepest. SCALE sets the resolution: peak slope is 1/(4*SCALE) per
# bit/byte, so 0.45 buys 0.556 units — 25 composite points — and a 0.1 bpb gain is
# worth ~2.5 points, comfortably clear of the reported stderr.
BPB_MID = 1.60
BPB_SCALE = 0.45

# Real-word rate saturates across any healthy cohort, so the *nonsense* rate is
# what carries information at this end. Every extra 10 nonsense words per hundred
# halves the component. Reaches 1.0 only at a perfect rate: an endpoint, not a clamp.
NONSENSE_HALVING = 0.10

# The repetition gate. Saturating at 1.0 is the design, not a defect: a healthy
# model must not be ranked on how far past "not looping" it is. Below FAIL the
# model is cycling and the other numbers have stopped describing language.
VARIETY_FAIL = 0.30
VARIETY_PASS = 0.70

W_LANGUAGE = 0.45
W_GRAMMAR = 0.40
W_WORDS = 0.15


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
def token_logprobs(model: Transformer, tokens: list[int], device: str = "cpu") -> TokenScores:
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


def language_component(bits_per_byte: float) -> float:
    return 1.0 / (1.0 + math.exp((bits_per_byte - BPB_MID) / BPB_SCALE))


def words_component(real_word_rate: float) -> float:
    return 2.0 ** (-(1.0 - real_word_rate) / NONSENSE_HALVING)


def variety_gate(distinct_3: float) -> float:
    # A missing measurement is not a failed one: no generated text is no evidence
    # of looping, so the gate opens.
    if not math.isfinite(distinct_3):
        return 1.0
    return _clamp((distinct_3 - VARIETY_FAIL) / (VARIETY_PASS - VARIETY_FAIL), 0.0, 1.0)


@dataclass
class ScoreBreakdown:
    """The score with its parts shown, so a number can be audited without
    re-deriving the formula — and a stored report re-scored under a later
    version."""

    version: int
    language: float
    grammar: float
    words: float
    variety_gate: float
    score: float
    score_stderr: float

    def as_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in vars(self).items()}


def composite(
    bits_per_byte: float,
    grammar_accuracy: float,
    grammar_pairs: int,
    real_word_rate: float,
    generated_words: int,
    distinct_3: float,
) -> ScoreBreakdown:
    """Blend the three components, redistributing the weight of any that are
    unavailable, then gate on repetition."""
    language = language_component(bits_per_byte) if math.isfinite(bits_per_byte) else math.nan

    # Grammar's lower clamp is a verdict, not a saturation: below chance is one
    # thing, not a scale.
    grammar = (
        _clamp((grammar_accuracy - 0.5) / 0.5, 0.0, 1.0)
        if math.isfinite(grammar_accuracy)
        else math.nan
    )
    words = words_component(real_word_rate) if math.isfinite(real_word_rate) else math.nan
    gate = variety_gate(distinct_3)

    components = [(W_LANGUAGE, language), (W_GRAMMAR, grammar), (W_WORDS, words)]
    present = sum(w for w, v in components if math.isfinite(v))

    if present <= 0.0:
        score = math.nan
    else:
        # Dividing by the *present* weight is the redistribution.
        blend = sum(w * v for w, v in components if math.isfinite(v)) / present
        score = 100.0 * blend * gate

    return ScoreBreakdown(
        version=SCORE_VERSION,
        language=language,
        grammar=grammar,
        words=words,
        variety_gate=gate,
        score=score,
        score_stderr=_stderr(grammar_accuracy, grammar_pairs, words, real_word_rate,
                             generated_words, present, gate),
    )


def _stderr(accuracy, pairs, words_value, real_word_rate, word_count, present, gate) -> float:
    """How far the score would move if the benchmark drew a different sample of
    the same size.

    Two independent estimation errors, propagated through their mappings and
    added in quadrature: grammar (binomial over the pair set, times the mapping's
    slope of 2) and words (binomial over the generated words, times the
    exponential's slope).

    **The language term is deliberately absent.** The held-out slice is a fixed
    set of bytes scored deterministically, and it is the *same* text for every
    model — its sampling error against "English at large" is a constant offset
    shared by every run, so it cancels in a model-to-model comparison. Inventing
    a number for it would inflate a gate and reject real improvements.
    """
    if present <= 0.0:
        return math.nan

    variance = 0.0
    if math.isfinite(accuracy) and pairs > 0:
        se = math.sqrt(max(accuracy * (1 - accuracy), 0.0) / pairs)
        variance += (W_GRAMMAR / present * 2.0 * se) ** 2
    if math.isfinite(real_word_rate) and word_count > 0 and math.isfinite(words_value):
        se = math.sqrt(max(real_word_rate * (1 - real_word_rate), 0.0) / word_count)
        slope = words_value * math.log(2) / NONSENSE_HALVING
        variance += (W_WORDS / present * slope * se) ** 2

    return 100.0 * gate * math.sqrt(variance)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
