# Telemetry — what a training run reports, and what it costs

A run publishes `Progress` frames (`llm/train/progress.py`); a sampled subset of
them carry a `diag` dict built by `llm/train/diagnostics.py`. This file is about
that dict: the budget it lives inside, the contract it travels under, and the
fine-tuning half of it that is absent on a pretrain run. The last section covers
the one frame that is not a `Progress` at all — phase 1's closing `Tokenize`, and
the span ribbon it carries.

## The budget is the design

`TARGET_REPORTS = 150` frames a run, whatever its length, and diagnostics are
sampled across those at `TARGET_DIAG_SAMPLES = 30`. Everything below happens on
a **diagnosed step only** — never per step — because it is measured against a
forward pass that costs real time and real memory.

Two properties hold that budget and both are load-bearing:

- **One recording pass, not two.** `Collector.observe` runs a single `no_grad`
  forward over `DIAGNOSTIC_ROWS` rows with the explicit-softmax attention path
  on. Every mask-derived reading is a *different weighting of the softmax that
  pass already computed*. A second forward for the loss splits would roughly
  double the diagnosed step.
- **No new host↔device sync on the training path.** `loop.run` drops all-zero
  SFT windows off the **host-side** mask (`weights.any()`), and `_LossLog`
  accumulates losses in a device buffer drained once per report. Nothing added
  here reads a device scalar outside a diagnosed step.

## The contract chain

```
Diagnostics -> Progress.diag -> api/runner._frame -> WebSocket
            -> frontend/lib/training.ts (DiagPoint)
            -> components/training/diagnostics-panel.tsx
```

**Field names are identical at every hop, on purpose.** Nothing in between
renames anything, so a rename in one place arrives at the chart as `undefined`.

**Everything on the wire is a scalar or a small fixed-length vector.** Nothing
scales with `seq^2` or with the vocabulary: the attention matrices and the
per-token logits are reduced to numbers before they leave the device.

**Unmeasured is absent, not zero.** `Diagnostics.as_dict` drops `None` fields,
the same convention `Tokenize` documents and for the same reason — a zero reads
as a real measurement of nothing, and a fine-tune whose base corpus could not be
reached would otherwise be indistinguishable from one whose loss on it is 0.

## The fine-tuning fields

Present only on an SFT run, and each independently absent when it could not be
reached. Their presence is how the panel tells a fine-tune from a pretrain —
there is no mode flag on a frame and there does not need to be one.

| Field | What it answers |
|---|---|
| `pretrain_loss` | Is the tune forgetting the language it came from? |
| `loss_masked` / `loss_unmasked` | Is it drifting on the prompt it only conditions on? |
| `loss_by_position` | Does it degrade toward the back of the context window? |
| `conversation_loss` / `conversation_start` | Which conversations cost it most? |
| `base_kl` | How far has it moved from the model it started as? |

Three things about them are decisions rather than implementation details:

- **The pretrain probe is reachable, and this is why.** A pretrain run records
  `corpus_key` in its `meta.json`, and the key names a directory under
  `datasets/pretrain/`. An SFT run already opens that file for
  `special_ids_seen`, so it can read the corpus the base was built from with no
  backend, no data engine, and no second tokenizer — the ids are the base
  tokenizer's on both sides, which is the property that makes it legitimate.
  It is held out from *this* run (SFT never trains on prose) but not from
  pretraining: read whether the number is climbing, not what it is.
- **A conversation is named by its corpus offset, never by a row index.** The
  epoch is reshuffled every pass (`EpochStream`), so a row index is a different
  conversation each epoch and a ranking built from one is an average over the
  dataset. `EpochStream.document_starts` answers this, and `iter_batches` yields
  each batch's offset in the epoch so the diagnostics can ask.
- **`base_kl` is opt-in (`SftConfig.measure_base_kl`, default off).** It is the
  one diagnostic that costs *memory* rather than time: the unmodified base model
  stays resident beside the one being trained. `orchestrate/sft.py` drops the
  reference the moment `loop.run` returns, because `loop.release_device` hands
  back the allocator's pool and not memory something still points at.

## Measured cost

At 512d/8h/6L/ctx512/bs8/vocab8192 on an RTX 3060, per **diagnosed** step, against
a 91.9 ms uncompiled ordinary step:

| | ms |
|---|---|
| `observe` as it was (pretrain shape) | 14.2 |
| + mask split, position buckets, per-conversation | 16.6 |
| + held-out pretrain loss | 25.3 |
| + `base_kl` (opt-in) | 57.9 |

So the always-on fine-tuning additions are ~11 ms on ~30 steps of a run — a
third of a second over a run of any length. Peak allocated during `observe` moves
from 692 to 724 MiB with the KL anchor on; the anchor's real cost is the second
copy of the weights it keeps resident, not that transient.

Pretrain runs pay **nothing**: `loop.run` passes no mask, `observe` skips every
split, and no probe is ever built.

## The span ribbon — phase 1's one non-scalar

`Tokenize.span_ribbon` is the exception to "everything on the wire is a scalar or
a small fixed-length vector", and it is worth stating why it earns the exception.

`boundary_straddle` is 0.63. That says the model attends across a document
boundary in most of its windows; it does not say what those windows *look like*,
and the two corpora it cannot tell apart want opposite decisions. A corpus of
books at 0.63 is windows sitting mid-chapter — one unbroken document, wider than
the window, and the straddle is measuring the chapter breaks. A corpus of tweets
at 0.63 is forty unrelated documents per window. Widen the window for the first;
reconsider the packing for the second.

So `spans.span_ribbon` samples actual windows and reports their layout:
`{datasets, windows: [{index, segments: [{dataset, start, length, opens, closes}],
overflow}]}`. Offsets are tokens from the window's first slot, clipped to it;
`opens`/`closes` are whether the document's own first and last token are inside,
which is the difference between a boundary the model attends across and an edge
that is only the window's.

Four things about it are decisions rather than implementation details:

- **The frame's size is a property of two constants, never of the corpus.**
  `RIBBON_WINDOWS = 48` windows, `RIBBON_SEGMENTS = 16` documents drawn per
  window, so `48 × 16` segments is the hard ceiling whatever the build weighs —
  9 KB of JSON for a corpus of long documents, 60 KB for the worst case where
  every window fills its cap. Same argument as `LENGTH_HIST_BINS` and
  `ACT_HIST_BINS`. A window past the cap reports `overflow` rather than more
  segments, so the renderer can draw the tail as the texture it is instead of
  claiming a long document where there are dozens of short ones.
- **The sample is evenly spaced, not the head of the corpus.** The first 48
  windows are the cheapest sample and the least representative one:
  `orchestrate.train._documents` yields stage by stage and dataset by dataset, so
  a build's opening windows are all one dataset and a head sample would show every
  multi-dataset corpus as homogeneous — not because it is, but because it is
  sorted. Spacing the sample gives each dataset a share of the strip in proportion
  to the tokens it contributed and keeps the strip's own order the corpus's.
- **Deterministic rather than random**, which outlives the figure: the run that
  packs a build and every later run that reuses it both report this frame, and two
  ribbons of one corpus that disagree read as the corpus having changed when only
  the sample did. The price is that a systematic sample can alias against a
  periodic corpus; packing is blocked by dataset rather than interleaved, so there
  is no period for the stride to lock onto.
- **It is the corpus as packed, not as a shuffled epoch reads it.** Windows are
  tiled from the corpus's first token, exactly as `window_fit` counts them, so the
  two figures describe one tiling. A run with `shuffle` on re-tiles the same
  documents in a different order, and `window_fit` argues — correctly — that this
  moves an *aggregate* by at most one window's worth of arrangement. That argument
  is weaker for a picture of specific windows: the *shapes* carry over (a corpus of
  books still gives unbroken bars, tweets still give confetti) but the adjacency
  and the dataset banding are the packing's, not the epoch's. Read the ribbon as
  what was packed. Showing what a shuffled epoch actually tiles would mean walking
  `EpochStream` for a seed and an epoch number, which is a different figure and a
  different frame.
- **It runs at report time, like `window_fit` and for the same reason.**
  `ctx_window` is not part of the corpus key — one build is legitimately trained
  at two window sizes — so nothing window-relative belongs in `segments.json`.
  Both are built in `train._phase_one`, which closes phase 1 for a fresh build and
  a reused one alike.

### Measured cost

At the 5B-token, 8.3M-document scale [`README.md`](README.md) quotes for
`window_fit`, reading the same mapped table:

| | ms | peak |
|---|---|---|
| `window_fit` | 263 | 41.9 MB |
| `span_ribbon` | 61 | 16.8 MB |

Cheaper than the function it illustrates, and for the reason that shapes the
implementation: `window_fit` histograms every row, where the ribbon only needs
two binary searches per sampled window per block. Both read the table in
`MEASURE_BLOCK` blocks, so peak memory is the block and not the build. A corpus
of short documents — the confetti case, where the inner walk actually runs — is
45 ms, because the walk is bounded by the sampled windows and not by the corpus.

Against a pack that is minutes to hours, that is noise. It is also paid once per
run rather than per step: this is a phase-1 frame, not a training-loop one.

The chain is the phase-1 chain rather than the `diag` one, and the same rule
holds at every hop:

```
spans.SpanRibbon -> train._phase_one -> Tokenize.span_ribbon
                 -> api/runner._frame -> WebSocket
                 -> frontend/lib/training.ts (Ribbon)
                 -> components/training/span-ribbon.tsx
```
