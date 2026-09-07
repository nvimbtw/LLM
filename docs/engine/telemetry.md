# Telemetry — what a training run reports, and what it costs

A run publishes `Progress` frames (`llm/train/progress.py`); a sampled subset of
them carry a `diag` dict built by `llm/train/diagnostics.py`. This file is about
that dict: the budget it lives inside, the contract it travels under, and the
fine-tuning half of it that is absent on a pretrain run. The last two sections
cover the frames that are not `Progress` at all — the `Tokenize` frames phase 1
publishes while it builds the data, and the span ribbon its closing one carries.

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

## What phase 1 says while it runs

Everything above is about a training *step*. Phase 1 is what happens before there
are any: a vocabulary learned, a corpus packed, and the measuring pass that closes
it. It used to publish three frames — the documents counted, the tokenizer
trained, and everything over — and the longest of its steps sat between the second
and the third saying nothing at all. At the 5B-token scale `README.md` quotes,
that is hours in which the only honest reading of the page was "no idea".

`BuildReport` is the seam that fixed it: made by the orchestrator, handed down
into the packer, and the only thing that turns a build's counters into frames.
Five things about it are decisions rather than implementation details.

- **The interval widens with the build, because the budget cannot be set in
  advance.** `TARGET_REPORTS` works for a training run because a run knows its
  step count before it takes a step. A build does not know whether packing is
  four minutes or four hours until it is over, and both are ordinary. So the gap
  between frames is `elapsed / TARGET_BUILD_REPORTS`, floored at
  `BUILD_REPORT_SECS`: a frame a second while a watcher is still deciding whether
  anything is happening, a frame a minute an hour in when the only question left
  is the ETA. The count then grows with the **logarithm** of the build's length —
  ~240 frames over four minutes, ~2,300 over four hours, against the ~14,000 a
  fixed one-second interval would leave in a run's history, which is held in
  memory for as long as anybody is attached.

- **The reporter travels down the layering; the frame type does not travel up.**
  `corpus` sits left of `train` in `llm/__init__.py`'s order, so `pack.pack`,
  `vocab.build` and `sft.build` take a `report` object rather than importing
  `Tokenize` — which is what `corpus/vocab.py` used to do, and was the one import
  in `llm/corpus` that reached right. A packer names the fields it is counting and
  knows nothing about what they become on the wire, so the names are still
  identical at every hop for the reason they always were: they are the ones it
  passed.

- **The step is named on the frame, never inferred from it.** The panel used to
  read "the step before the first one that has not reported yet is the one
  running", which is true only while every step reports something. It also meant
  a step could only end the phase by *saying* so — and `vocab.build` said so,
  which for a pretrain is the first tenth of phase 1. `done=True` sent the panel
  to the training view for the whole of the pack that followed, which is the
  proximate cause of the silence this section is about. `BuildReport.finish` is
  now the only thing that can close the phase, and only an orchestrator calls it.

- **The bar is a table of spans, not a division into equal steps.** The steps are
  fixed and their costs are wildly uneven — packing is hours where learning merges
  is seconds — so equal shares would leave the bar at a third for all but the last
  minute of a build. `BUILD_SPANS` gives each step a cumulative span and every
  step reports its own fraction *within* that span; a build enters them in order,
  which is what makes `progress` monotone without anything having to clamp it.

- **Two rates, because they answer different questions.** `eta_secs` is projected
  from the rate over the whole step, which is the smooth one and the same
  estimator `train.loop` gives a run. `tokens_per_sec` is over the last
  `RATE_WINDOW_SECS`, which is the one that says whether the thing is moving
  *now*. An average would take minutes to show a stall and minutes more to
  forgive one. Both are `BuildReport.pace`, so a pack and an SFT render cannot
  disagree about what either means.

- **What a pack divides by is text, not documents.** `pace` takes whatever unit
  it is handed, and the packer hands it characters against the plan's
  `expected_chars`. A document is not an amount of text — a selection of books
  and web snippets is 86% books by bytes and 5.6% by documents — so a pack a
  quarter of the way through the books had read 3% of the documents it was asked
  for and reported fourteen hours on a build that takes under one. The bar moves
  in the same unit, for the same reason. See
  [`corpus.md`](corpus.md).

| Step | Reports |
|---|---|
| `sampling` | `sampled_bytes` against `sample_budget` — the text the merges will be fitted to |
| `merges` | nothing while it runs: one call into `tokenizers`, which exposes no callback (the same fact `merge_sequence` is shaped by). The step's name and the clock are the whole of what a watcher gets, and they are the difference between working and hung |
| `packing` | `documents_done`/`documents`, `chars`, `total_tokens`, and `sources`. The bar and the ETA are `chars` against the plan's, not documents against documents |
| `rendering` | a fine-tune's one long step: `documents_done`, `conversations`, `total_tokens`, `trainable_tokens`, `skipped_too_long` |
| `reusing` | that the build was found on disk, so the hours are skipped and the reporting is not |
| `measuring` | the closing frame — `window_fit`, the ribbon, fertility, dead vocab |

`sources` is the one field that is not a scalar, and it is bounded the way
`fertility` is: `pack.BUILD_SOURCES` rows of `{dataset, stage, expected,
expected_chars, documents, chars, tokens}`, with `sources_total` carrying the
real count. `expected_chars` is absent rather than zero on a plan that did not
state it, which is what decides the unit above. It is
keyed by `(stage, dataset)` and not by dataset alone, because a dataset two
stages name is read twice on purpose and a bar that summed the two passes would
sit at half through the whole of the first. The window is **centred on the source
being read** rather than taken from the front, which is the difference between
showing where a build has got to and showing where it started.

### Measured cost

Nothing. Every counter a frame carries is one the pack already kept — the
per-dataset characters and tokens are the sidecar's own `fertility`, computed at
the one moment the text and its encoding are both in hand — and the only per-frame
work is copying at most `BUILD_SOURCES` small dicts. The throttle is checked
before any of that is assembled, so a build with nobody attached pays a clock
read per encode batch.

The chain is the phase-1 chain, and the same rule holds at every hop:

```
BuildReport -> Tokenize -> api/runner._frame -> WebSocket
            -> frontend/lib/training.ts (TokenizeStats, BuildSource)
            -> components/training/build-progress.tsx
```

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
