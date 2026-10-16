"""A small dual-kind processor demonstrating operator file-drop registration."""

import re

from processors.helpers import Ctx, Param, processor

URL_RE = re.compile(r"https?://\S+|www\.\S+")


@processor(
    name="strip-urls",
    title="Strip URLs",
    kinds=["pretrain", "sft"],
    params=[
        Param(name="min_len", label="Min length", type="int", default=0, minimum=0)
    ],
    mode="text",
)
def run(text: str, p: dict, ctx: Ctx) -> str | None:
    """Remove URLs; drop empty or too-short text, including its containing exchange."""
    stripped = URL_RE.sub("", text)
    if not stripped.strip() or len(stripped) < p["min_len"]:
        return None
    return stripped
