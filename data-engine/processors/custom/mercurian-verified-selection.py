import json
from processors.helpers import Ctx, Param, processor


@processor(name="mercurian-verified-selection", title="Mercurian tokenizer-verified selection", kinds=["sft"], mode="document", description="Keep only explicitly reviewed candidate ordinals and record their measured token counts. The allowlist must be generated with the intended base tokenizer, not character estimates.", params=[Param(name="verified", label="JSON ordinal to measured token count", type="string", default="{}"), Param(name="base_model", label="Tokenizer base model UUID", type="string", default="339d57f5-8442-416d-8da7-2b70d43624f0"), Param(name="max_tokens", label="Maximum full exchange tokens", type="int", default=1024, minimum=1, maximum=1024)])
def run(document: dict, params: dict, ctx: Ctx) -> dict | None:
    """Select sequential input ordinals measured externally and annotate provenance.

    Counts must include the chat template and use separate training segments.
    This processor does not tokenize or guarantee a different base model fits.
    """
    key = "mercurian-verified-selection"
    if key not in ctx.state:
        verified = json.loads(params["verified"])
        if not isinstance(verified, dict):
            raise ValueError("verified must be an ordinal-to-token-count object")
        if any(not str(index).isdigit() or type(count) is not int or count < 1 for index, count in verified.items()):
            raise ValueError("verified entries must be nonnegative ordinals and positive integer counts")
        ctx.state[key] = {"next": 0, "verified": verified}
    state = ctx.state[key]
    ordinal = state["next"]
    state["next"] += 1
    tokens = state["verified"].get(str(ordinal))
    if tokens is None or tokens > params["max_tokens"]:
        return None
    meta = {**(document.get("meta") or {}), "verified_candidate_ordinal": ordinal, "verified_full_tokens": tokens, "verified_tokenizer_model": params["base_model"]}
    return {**document, "meta": meta}
