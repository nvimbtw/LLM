"""What a diagnosed step is allowed to hold on to.

`Collector.observe` records an explicit attention softmax, which materializes a
`[rows, heads, seq, seq]` fp32 matrix per layer where the fused kernel
materializes nothing. Those matrices used to live on the blocks from the
recording forward until `after_step` walked them — and `after_step` necessarily
runs after the optimizer, so every one of them was alive across the whole
training step it was measuring.

That is quadratic in sequence length and linear in depth, which is invisible at
a short context and decisive at a long one. Measured on an 18-layer model at
`DIAGNOSTIC_ROWS = 2`:

```
ctx 512    ordinary 2419 MiB    diagnosed 2743 MiB   1.13x
ctx 1024   ordinary 3743 MiB    diagnosed 4967 MiB   1.33x
ctx 2048   ordinary 6385 MiB    diagnosed  OOM
```

Reducing each matrix to its per-head entropy as the forward walks past it — the
hook `Collector.__enter__` installs — leaves one matrix alive instead of
`layers` of them, and the same three rows read 1.015x, 1.019x and 1.025x.

So this file is about *absence*: nothing the size of an attention matrix may
survive the pass that made it. The numbers above are what the assertion is
worth, and `tests/test_attention.py` is a different question (the route, and the
flag it must put back).
"""

import numpy as np
import torch

from llm.model.transformer import Transformer
from llm.train.diagnostics import Collector, _entropy, diagnostic_rows

VOCAB, DIMS, HEADS, LAYERS, CTX = 128, 32, 4, 5, 16


def _collector() -> tuple[Collector, Transformer]:
    model = Transformer(VOCAB, DIMS, HEADS, LAYERS, 4, CTX)
    model.train()
    # `counts` rather than a corpus, so the collector's token ranking does not
    # have to read one. Either is legal; this one is cheaper.
    counts = np.arange(VOCAB, dtype=np.int64)[::-1].copy()
    return Collector(model, None, counts, None), model


def test_no_attention_matrix_survives_the_recording_pass():
    """The whole point: `after_step` finds nothing left to walk.

    A matrix still on a block here is one that would have been alive across the
    training step — 4.8 GiB of them on an 18-layer model at ctx 2048.
    """
    collector, model = _collector()
    inputs = torch.randint(0, VOCAB, (4, CTX))

    collector.observe(inputs)

    assert all(block.attn.last_attention is None for block in model.blocks)


def test_every_layer_is_still_measured():
    """Freeing them early must not mean losing them — one reading per layer."""
    collector, model = _collector()
    inputs = torch.randint(0, VOCAB, (4, CTX))

    collector.observe(inputs)

    assert len(collector._entropy) == LAYERS
    assert all(len(heads) == HEADS for heads in collector._entropy)
    assert all(0.0 <= value <= 1.0 for heads in collector._entropy for value in heads)


def test_the_reading_is_the_one_the_walk_would_have_taken():
    """Same arithmetic, taken earlier — the hook reduces, it does not approximate.

    Recorded a second time by hand, off the same weights and the same rows, and
    compared against what the collector kept.
    """
    collector, model = _collector()
    inputs = torch.randint(0, VOCAB, (4, CTX))

    collector.observe(inputs)
    measured = collector._entropy

    rows = diagnostic_rows(inputs.shape[0])
    for block in model.blocks:
        block.attn.record_attention = True
    with torch.no_grad():
        model(inputs[:rows])
    expected = [_entropy(block.attn.last_attention) for block in model.blocks]

    assert measured == expected


def test_the_hooks_do_not_outlive_the_context():
    """Left installed, they would reduce and free on every later forward.

    `observe`'s own probe losses run outside the recording context, and so does
    every chat turn in a process that has trained — a hook still attached would
    be doing diagnostics work on all of them.
    """
    collector, model = _collector()
    inputs = torch.randint(0, VOCAB, (4, CTX))

    collector.observe(inputs)

    assert collector._handles == []
    assert all(not block.attn._forward_hooks for block in model.blocks)


def test_a_collector_that_never_recorded_reports_nothing():
    """`after_step` off an unused collector reads empty, rather than raising."""
    collector, _ = _collector()
    assert collector._entropy == []
