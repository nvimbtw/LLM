"""The markup suite: does the model use the marker grammar correctly?

Four things, weighted: it prefers well-formed markup over malformed (pairs), it
puts a marker where one belongs (positions), what it generates parses and ends
its own turn (structure), and it does not emit markers where prose belongs
(restraint).

**Generation here is deliberately unconstrained.** `llm.chat.markup` is a
serving-time constraint; generating under it would measure the constraint and
score every model alike. This suite exists to say whether the model has *learned*
the grammar.

Ported from backend/crates/llm/src/eval/markup.rs.
"""

import math
from dataclasses import dataclass, field

import torch
from tokenizers import Tokenizer

from llm.chat.markup import PAIRS, Rules
from llm.chat.template import Reasoning, Role, ThinkBudget, Turn, build_training_example
from llm.corpus import bpe
from llm.eval import score as score_mod
from llm.model.transformer import Transformer

MARKUP_SCORE_VERSION = 1

W_PAIRS = 0.40
W_POSITIONS = 0.30
W_STRUCTURE = 0.20
W_RESTRAINT = 0.10

# A reply containing one of these is impersonating the other speaker or reopening
# a turn it is still inside. It corrupts the transcript whether or not the markup
# around it balances.
SCAFFOLD_MARKERS = [
    "<|user|>", "<|/user|>", "<|system|>", "<|/system|>", "<|assistant|>",
]


def ids_are_balanced(ids: list[int], rules: Rules) -> bool:
    """Is this id sequence's markup balanced and properly nested?

    Over ids rather than decoded text: every marker is exactly one reserved id,
    so there is no scanning, no ambiguity about text that merely looks like a
    marker, and no dependence on the decoder round-tripping.
    """
    stack: list[int] = []
    for token_id in ids:
        if token_id in rules.closer_for:
            stack.append(token_id)
        elif (opener := rules.opener_for.get(token_id)) is not None:
            if not stack or stack.pop() != opener:
                return False
    return not stack


@dataclass
class PositionMetrics:
    # The scored one: at a reply position whose reference next token is a marker,
    # does the model's argmax over the *whole vocabulary* agree? A model can give
    # a marker respectable log-probability while confidently predicting something
    # else, and generation would then not put the marker there.
    marker_top1: float
    # Argmax restricted to the marker block. Separates "picks the right marker"
    # from "knows a marker belongs here" — the two failures want different fixes.
    marker_restricted_top1: float
    marker_mean_rank: float
    marker_logprob: float
    marker_positions: int
    # Over reply positions whose reference next token is *not* a marker: how often
    # the model wanted to emit one anyway.
    false_marker_rate: float
    prose_positions: int
    conversations: int

    def as_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in vars(self).items()}


@dataclass
class StructureMetrics:
    # Did the reply end its own turn rather than running to the token budget?
    close_rate: float
    # Was the emitted marker sequence balanced and properly nested?
    balance_rate: float
    # Markers per 100 generated tokens, against the same figure over the held-out
    # replies. Reported, not scored: it catches both "emits none" and "floods",
    # and which of those is bad depends on the reference.
    marker_density: float
    reference_marker_density: float
    scaffold_leak_rate: float
    replies: int
    tokens_generated: int
    samples: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in vars(self).items()}


def score_positions(
    model: Transformer,
    tokenizer: Tokenizer,
    conversations: list[dict],
    rules: Rules,
    reasoning: Reasoning = Reasoning.KEEP,
    device: str = "cpu",
) -> PositionMetrics | None:
    """At each reply position, compare the model's prediction to the reference."""
    markers = set(rules.marker_ids)
    marker_index = torch.tensor(sorted(markers), device=device)

    top1 = restricted = 0
    marker_positions = 0
    rank_total = 0.0
    logprob_total = 0.0
    false_markers = 0
    prose_positions = 0
    scored_conversations = 0

    for conversation in conversations:
        turns: list[Turn] = []
        bad = False
        for turn in conversation.get("turns", []):
            if turn.get("role") not in ("user", "assistant"):
                bad = True
                break
            turns.append(Turn(role=Role(turn["role"]), text=turn.get("text", "")))
        if bad or not turns:
            continue

        budget = ThinkBudget.parse(conversation["budget"]) if conversation.get("budget") else None
        ids, mask = build_training_example(
            conversation.get("system"), turns, tokenizer, budget, reasoning
        )
        if len(ids) < 2:
            continue

        with torch.inference_mode():
            window = ids[: model.ctx_window]
            logits, _ = model(torch.tensor([window], device=device))
            log_probs = torch.log_softmax(logits[0].float(), dim=-1)

        scored_conversations += 1
        for i in range(len(window) - 1):
            if not mask[i]:
                continue
            reference = window[i + 1]
            row = log_probs[i]
            predicted = int(row.argmax())

            if reference in markers:
                marker_positions += 1
                if predicted == reference:
                    top1 += 1
                if int(marker_index[row[marker_index].argmax()]) == reference:
                    restricted += 1
                # Rank of the correct marker over the whole vocabulary.
                rank_total += float((row > row[reference]).sum()) + 1.0
                logprob_total += float(row[reference])
            else:
                prose_positions += 1
                if predicted in markers:
                    false_markers += 1

    if marker_positions == 0 and prose_positions == 0:
        return None

    def ratio(n, d):
        return n / d if d else math.nan

    return PositionMetrics(
        marker_top1=ratio(top1, marker_positions),
        marker_restricted_top1=ratio(restricted, marker_positions),
        marker_mean_rank=ratio(rank_total, marker_positions),
        marker_logprob=ratio(logprob_total, marker_positions),
        marker_positions=marker_positions,
        false_marker_rate=ratio(false_markers, prose_positions),
        prose_positions=prose_positions,
        conversations=scored_conversations,
    )


def reference_marker_density(
    tokenizer: Tokenizer, conversations: list[dict], rules: Rules
) -> float:
    """Markers per 100 reply tokens in the held-out set — what the model's own
    density is compared against."""
    markers = set(rules.marker_ids)
    total = marked = 0
    for conversation in conversations:
        for turn in conversation.get("turns", []):
            if turn.get("role") != "assistant":
                continue
            ids = bpe.encode(tokenizer, turn.get("text", ""))
            total += len(ids)
            marked += sum(1 for i in ids if i in markers)
    return 100.0 * marked / total if total else math.nan


def composite_score(
    pair_accuracy: float,
    pair_count: int,
    positions: PositionMetrics | None,
    structure: StructureMetrics | None,
) -> dict:
    """Blend the four components, redistributing the weight of any unavailable."""
    # Rescaled from the 0.5 chance floor, like grammar: a model that guesses
    # scores 0.
    pairs = (
        max(0.0, min(1.0, (pair_accuracy - 0.5) / 0.5))
        if math.isfinite(pair_accuracy)
        else math.nan
    )
    position_value = positions.marker_top1 if positions else math.nan
    # Three equally weighted halves-of-a-whole: a reply that never ends, one whose
    # markup does not parse, and one that impersonates the user are all unusable,
    # and there is no reason to rank those failures against each other.
    structure_value = (
        (structure.close_rate + structure.balance_rate + (1.0 - structure.scaffold_leak_rate)) / 3.0
        if structure
        else math.nan
    )
    restraint = (1.0 - positions.false_marker_rate) if positions else math.nan

    components = [
        (W_PAIRS, pairs),
        (W_POSITIONS, position_value),
        (W_STRUCTURE, structure_value),
        (W_RESTRAINT, restraint),
    ]
    present = sum(w for w, v in components if math.isfinite(v))
    score = (
        100.0 * sum(w * v for w, v in components if math.isfinite(v)) / present
        if present > 0
        else math.nan
    )

    def clean(v):
        return None if isinstance(v, float) and math.isnan(v) else v

    return {
        "version": MARKUP_SCORE_VERSION,
        "pairs": clean(pairs),
        "positions": clean(position_value),
        "structure": clean(structure_value),
        "restraint": clean(restraint),
        "score": clean(score),
        "pair_count": pair_count,
    }
