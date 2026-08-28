"""One head's attention over one piece of text, materialized when asked for.

Nothing here records anything by default, and that is the design rather than an
omission. At 512d / 8 heads / 6 layers / ctx 512 the full attention tensor is
~8 MiB a layer and ~50 MiB a sample, so a run that streamed it would carry about
1.5 GiB of it — see `../train/diagnostics.py` for what merely *computing* it on
the training forward already cost. One `(layer, head)` matrix at ctx 512 is
512x512 fp32, exactly 1 MiB, which is affordable precisely because a request is
what brings it into existence and nothing else does.

`Attention.record_attention` is the mechanism, unchanged: the same flag the
diagnostics collector flips, flipped here for a single forward pass and flipped
back in a `finally`. There is deliberately no second recording path.
"""

import base64

import torch

from llm.corpus import bpe
from llm.model.transformer import Attention
from llm.orchestrate import chat

# The longest text a request may inspect, in tokens.
#
# The matrix is quadratic in this number, so this — not a byte ceiling — is the
# thing that has to be bounded: 512 tokens is 1 MiB of fp32 on the wire, 1024 is
# 4 MiB, and 4096 would be 64 MiB out of a route anyone signed in can call. A
# model whose own `ctx_window` is smaller binds first; this is the ceiling for
# everything above it.
MAX_SEQUENCE = 512

# The wire encoding, stated once so the response can name it.
#
# Cost at the dimensions this project actually trains: a 512x512 fp32 matrix is
# 1,048,576 bytes raw and 1,398,104 base64 — call it 1.33 MiB. The same numbers
# as JSON are 2.5-3 MiB rounded to six decimals and roughly 5 MiB unrounded,
# because a float costs 8-20 characters and two more for the comma. So base64 is
# ~2x the best JSON can do and needs no decision about how much precision to
# throw away to get there.
#
# fp16 would halve it again. It was not taken: the browser has no native half
# decoder, so it buys 700 KiB on a request a person makes by hand at the price of
# a hand-rolled decoder, and it puts "the rows sum to 1" inside a tolerance
# instead of at the exactness the model computed it to.
WIRE_DTYPE = "float32"
_NUMPY_DTYPE = "<f4"


def matrix(model_name: str, text: str, layer: int, head: int) -> dict:
    """The `(layer, head)` attention matrix over `text`, with its axis labels.

    Raises `FileNotFoundError` for a model with no weights and `ValueError` for
    a layer, a head, or a length this model cannot answer for.
    """
    model, tokenizer, device = chat.load(model_name)

    _check_index("layer", layer, model.layers)
    _check_index("head", head, model.heads)

    ids = bpe.encode(tokenizer, text)
    limit = min(MAX_SEQUENCE, model.ctx_window)
    if not ids:
        raise ValueError("text encodes to no tokens")
    if len(ids) > limit:
        raise ValueError(
            f"text is {len(ids)} tokens; this model's attention can be inspected "
            f"up to {limit} (the matrix is quadratic in that number)"
        )

    weights = _record_one_head(model, torch.tensor([ids], device=device), layer, head)

    return {
        "model_name": model_name,
        "layer": layer,
        "head": head,
        "layers": model.layers,
        "heads": model.heads,
        "max_tokens": limit,
        "tokens": [bpe.token_text(tokenizer, tid) for tid in ids],
        "shape": list(weights.shape),
        "dtype": WIRE_DTYPE,
        "encoding": "base64",
        # Row-major: row i is what the token at position i attended to, which is
        # the direction the causal mask runs and the direction a reader reads.
        "weights": base64.b64encode(weights.tobytes()).decode("ascii"),
    }


def _check_index(name: str, value: int, count: int) -> None:
    if not 0 <= value < count:
        raise ValueError(f"{name} {value} is out of range; this model has {count}")


def _record_one_head(model, ids: torch.Tensor, layer: int, head: int):
    """One forward with recording on for exactly one block, off again after.

    Only the asked-for block records. The flag is per-`Attention`, so the other
    layers keep the fused kernel and never materialize a matrix nothing reads —
    which is the same argument `Collector` makes about the training forward, one
    layer smaller.

    The model is the one `chat.load` caches, so a chat generating in another
    thread during this window takes the explicit-softmax path for one block. It
    is the same arithmetic (`bench.py` pins the two to 1e-5) and it is slower;
    the alternative is a second resident copy of the weights, which is the cost
    that cache exists to avoid.
    """
    block = model.blocks[layer]
    attention: Attention = block.attn
    attention.record_attention = True
    try:
        with torch.no_grad():
            model(ids)
        recorded = attention.last_attention
    finally:
        # In a `finally` because a forward that raises — a bad length, an OOM —
        # would otherwise leave the flag on and quietly move every later chat
        # turn off the fused kernel.
        attention.record_attention = False
        attention.last_attention = None

    if recorded is None:
        raise ValueError(f"layer {layer} recorded no attention")
    # [batch, heads, seq, seq] -> [seq, seq], fp32 on the host. Contiguous
    # because `tobytes` on a view of a slice would otherwise copy anyway, and
    # this way the copy is named.
    return recorded[0, head].detach().float().cpu().numpy().astype(_NUMPY_DTYPE, copy=False)
