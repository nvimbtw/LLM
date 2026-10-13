"""The assistant suite: how well does the model predict a held-out reply?

Scored through the same loss mask SFT trains on, so the positions measured here
are exactly the positions a fine-tune would have taken gradient on.

Ported from backend/crates/llm/src/eval/chat.rs.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

from llm.chat.template import (
    Reasoning,
    Role,
    ThinkBudget,
    Turn,
    build_training_example,
)
from llm.corpus import bpe
from llm.model import likelihood as score_mod
from llm.model.transformer import Transformer

# Bump alongside any change to the anchors below: it moves every score without a
# model having changed.
#
# **Version 2 changed the unit**, from bits per reply token to bits per reply
# byte. Nothing recorded at version 1 is comparable, and the gap is not a
# rescaling that could be undone arithmetically — a v1 score depends on the
# bytes-per-token of the tokenizer it was measured under, which the report does
# not record.
ASSISTANT_SCORE_VERSION = 2

# Bits per reply *byte* through a logistic — same shape as the language suite's
# constants, deliberately NOT shared with them. Chat replies carry markup and
# prose does not, so the two distributions are not the same distribution, and one
# suite's calibration drifting into the other's is how they would stop being
# independently versioned.
CHAT_BPB_MID = 1.60
CHAT_BPB_SCALE = 0.45


@dataclass
class ReplyScore:
    """One conversation's contribution: total log-probability over its reply
    tokens, and the two denominators that turn it into a rate.

    `bytes` is the UTF-8 length of what those tokens decode to. Carried rather
    than derived later because it is only recoverable while the ids are in hand.
    """

    logprob: float
    tokens: int
    bytes: int


@dataclass
class ChatMetrics:
    # The headline. Per byte rather than per token because a token is not a fixed
    # unit: a tokenizer change moves every bits-per-token figure on record even
    # on unchanged weights. A byte is the same byte under any tokenizer.
    bits_per_byte: float
    bits_per_byte_stderr: float
    bytes_scored: int
    # Reported but no longer scored. Kept because it is the masked cross-entropy
    # SFT actually minimizes, so it is the number to read when asking whether a
    # fine-tune's own objective moved. Comparable only between models sharing a
    # tokenizer.
    bits_per_token: float
    bits_per_token_stderr: float
    logprob: float
    conversations: int
    tokens_scored: int

    def as_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in vars(self).items()}


def load(directory: Path) -> list[dict]:
    """Held-out conversations, one JSON object per line.

    A malformed line is skipped rather than failing the run: this directory is
    hand-maintained, and one bad line should cost its own conversation, not the
    whole benchmark.
    """
    if not directory.is_dir():
        return []
    conversations: list[dict] = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                conversations.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return conversations


def reply_logprob(
    model: Transformer,
    tokenizer: Tokenizer,
    conversation: dict,
    reasoning: Reasoning = Reasoning.KEEP,
    device: str = "cpu",
) -> ReplyScore | None:
    """Total log-probability the model assigns to a conversation's reply tokens."""
    turns: list[Turn] = []
    for turn in conversation.get("turns", []):
        role = turn.get("role", "")
        if role not in ("user", "assistant"):
            # An unknown role would silently shift the mask onto the wrong
            # tokens, so the conversation is refused rather than guessed at.
            return None
        turns.append(Turn(role=Role(role), text=turn.get("text", "")))
    if not turns:
        return None

    budget = ThinkBudget.parse(conversation["budget"]) if conversation.get("budget") else None

    # The same rendering SFT trains on.
    ids, mask = build_training_example(
        conversation.get("system"), turns, tokenizer, budget, reasoning
    )
    if len(ids) < 2:
        return None

    scores = score_mod.token_logprobs(model, ids, device)
    if scores.count == 0:
        return None

    # `token_logprobs` scores positions 1..n; mask[i] marks position i as
    # predicting a reply token, so mask[i] gates the score of token i+1.
    total = 0.0
    scored_ids: list[int] = []
    for i, logprob in enumerate(scores.logprobs):
        if mask[i]:
            total += logprob
            scored_ids.append(ids[i + 1])

    if not scored_ids:
        return None

    return ReplyScore(
        logprob=total,
        tokens=len(scored_ids),
        bytes=len(bpe.decode(tokenizer, scored_ids).encode("utf-8")),
    )


def _ratio_stderr(pairs: list[tuple[float, int]], rate: float, total_units: int) -> float:
    """Cluster-robust standard error of a ratio of two sums.

    The sampling unit is the **conversation**, not the byte. Bytes within one
    reply are nothing like independent — a reply that goes wrong goes wrong for
    fifty tokens together — so a binomial over bytes would understate the error
    several-fold and a gate would read noise as progress.
    """
    n = len(pairs)
    if n < 2 or total_units == 0:
        return math.nan
    sum_squares = sum((bits - rate * units) ** 2 for bits, units in pairs)
    return math.sqrt(sum_squares * n / (n - 1)) / total_units


def summarize(scored: list[ReplyScore]) -> ChatMetrics | None:
    if not scored:
        return None

    total_logprob = sum(s.logprob for s in scored)
    total_tokens = sum(s.tokens for s in scored)
    total_bytes = sum(s.bytes for s in scored)
    if total_tokens == 0 or total_bytes == 0:
        return None

    total_bits = -total_logprob / math.log(2)
    bits_per_byte = total_bits / total_bytes
    bits_per_token = total_bits / total_tokens

    per_conversation_bits = [(-s.logprob / math.log(2)) for s in scored]

    return ChatMetrics(
        bits_per_byte=bits_per_byte,
        bits_per_byte_stderr=_ratio_stderr(
            list(zip(per_conversation_bits, (s.bytes for s in scored))),
            bits_per_byte,
            total_bytes,
        ),
        bytes_scored=total_bytes,
        bits_per_token=bits_per_token,
        bits_per_token_stderr=_ratio_stderr(
            list(zip(per_conversation_bits, (s.tokens for s in scored))),
            bits_per_token,
            total_tokens,
        ),
        logprob=total_logprob / total_tokens,
        conversations=len(scored),
        tokens_scored=total_tokens,
    )


def assistant_score(metrics: ChatMetrics | None) -> dict:
    """The assistant composite: one mapped component, so the score IS the
    component times 100."""
    if metrics is None or not math.isfinite(metrics.bits_per_byte):
        return {
            "version": ASSISTANT_SCORE_VERSION,
            "replies": None,
            "score": None,
            "score_stderr": None,
        }

    replies = 1.0 / (1.0 + math.exp((metrics.bits_per_byte - CHAT_BPB_MID) / CHAT_BPB_SCALE))
    # Propagated from bits_per_byte_stderr through the logistic's slope,
    # p*(1-p)/SCALE, so a gate can be derived here the same way as elsewhere.
    stderr = metrics.bits_per_byte_stderr
    score_stderr = (
        100.0 * replies * (1.0 - replies) / CHAT_BPB_SCALE * stderr
        if math.isfinite(stderr)
        else None
    )
    return {
        "version": ASSISTANT_SCORE_VERSION,
        "replies": replies,
        "score": 100.0 * replies,
        "score_stderr": score_stderr,
    }
