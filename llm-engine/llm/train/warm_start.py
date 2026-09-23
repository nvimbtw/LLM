"""Warm-starting embedding rows before a fine-tune.

A marker absent from the pretrain corpus reaches SFT with its embedding still at
random init, and injects noise into the shared trunk the instant SFT feeds it —
at the usual SFT learning rate, wrecking the pretrained weights. That is the
"SFT turns coherent output into gibberish" failure. Resetting the row to the mean
embedding (a "typical token" vector) is the standard added-token initialization.

**The caller supplies the ids and owns the policy.** This deliberately does not
*detect* untrained rows, because that is not decidable from the weights. The Rust
version tried — "all components within the init bound" — and on a small model
that bound catches around half of all legitimately-trained-but-small embeddings
(rare bytes, infrequent merges, lightly-used markup), so it reset half the
vocabulary onto one mean vector and *caused* the gibberish it was meant to
prevent. A weight value does not carry its own history; only a record made at
pretrain time does, which is what `special_ids_seen` is for.
"""

import torch

from llm.model.transformer import Transformer


def warm_start_embeddings(model: Transformer, ids: list[int]) -> int:
    """Reset each row in `ids` to the mean embedding. Returns how many were reset.

    Ids at or past the vocabulary are skipped rather than rejected: a mismatched
    record should not fail a run that is otherwise fine.
    """
    if not ids:
        return 0

    with torch.no_grad():
        table = model.embedding.weight
        mean = table.mean(dim=0)
        reset = 0
        for token_id in ids:
            if 0 <= token_id < table.shape[0]:
                table[token_id] = mean
                reset += 1
    return reset


def untrained_marker_ids(used: list[int], seen: list[int] | None) -> list[int]:
    """Markers this dataset uses that the base model's pretrain corpus never had.

    `seen` of None means "unknown" — a model trained before the record existed —
    and yields an empty list rather than assuming nothing was seen, which would
    reset every marker the fine-tune touches.
    """
    if seen is None:
        return []
    return sorted(set(used) - set(seen))
