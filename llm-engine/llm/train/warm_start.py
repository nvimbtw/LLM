"""Capability-owned token-table warm-start before a fine-tune, with a deliberate legacy adapter.

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

Declared custom capabilities own their tied/untied input and output tables.
Unsupported whole models retain all weights and return a distinct skipped status.
"""

from torch import nn

from llm.architecture.build import RecipeModule
from llm.architecture.interfaces import TokenRowInitializer, TokenRowResult
from llm.architecture.plan import PlanCall
from llm.model.transformer import Transformer


def declared_initializer(model: nn.Module) -> TokenRowInitializer | None:
    """Find an explicitly declared whole-model token-table owner, without inferring ownership from tensor names.

    A composition delegates only when its sole child is called once with the
    complete unchanged named interface. Loops, other children and tensor-only
    compositions do not promote a local block capability to the whole model.
    """
    if isinstance(model, RecipeModule):
        if model.plan.interface != "causal_lm" or len(model.instances) != 1 or len(model.plan.operations) != 1:
            return None
        call = model.plan.operations[0]
        if (not isinstance(call, PlanCall) or call.inputs != {name: name for name in model.plan.inputs}
                or call.outputs != model.plan.result):
            return None
        return declared_initializer(model.instances[call.instance])
    if "token_rows" in getattr(model, "__slate_capabilities__", ()) and isinstance(model, TokenRowInitializer):
        return model
    return None


def initialize_token_rows(model: nn.Module, ids: list[int]) -> TokenRowResult:
    """Initialize unique requested rows through declared table ownership, or skip without changing unsupported weights.

    The caller owns an admitted worker and supplies marker provenance; weights
    cannot reveal which markers were trained. Invalid ID types/budgets refuse
    before mutation. Out-of-vocabulary IDs follow the capability's skipped report.
    """
    if len(ids) > 65536 or any(type(token_id) is not int for token_id in ids):
        raise ValueError("token row initialization requires at most 65536 integer IDs")
    token_ids = tuple(dict.fromkeys(ids))
    if isinstance(model, Transformer):
        return model.initialize_token_rows(token_ids)
    initializer = declared_initializer(model)
    if initializer is None:
        return TokenRowResult("skipped_unsupported", skipped_ids=token_ids)
    result = initializer.initialize_token_rows(token_ids)
    if (not isinstance(result, TokenRowResult) or type(result.initialized_ids) is not tuple
            or type(result.skipped_ids) is not tuple):
        raise ValueError("token row capability returned an invalid outcome")
    reported = result.initialized_ids + result.skipped_ids
    if (result.status not in {"initialized", "skipped_unsupported"}
            or any(type(token_id) is not int for token_id in reported)
            or len(reported) != len(set(reported)) or set(reported) != set(token_ids)
            or result.status == "skipped_unsupported" and result.initialized_ids):
        raise ValueError("token row capability returned inconsistent row ownership")
    return result


def warm_start_embeddings(model: nn.Module, ids: list[int]) -> int:
    """Keep the existing count-returning SFT call compatible while routing through explicit token-row ownership."""
    return len(initialize_token_rows(model, ids).initialized_ids)


def untrained_marker_ids(used: list[int], seen: list[int] | None) -> list[int]:
    """Markers this dataset uses that the base model's pretrain corpus never had.

    `seen` of None means "unknown" — a model trained before the record existed —
    and yields an empty list rather than assuming nothing was seen, which would
    reset every marker the fine-tune touches.
    """
    if seen is None:
        return []
    return sorted(set(used) - set(seen))
