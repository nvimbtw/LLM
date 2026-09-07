# The corpus a run trains on — where it comes from, what it is made of

Split out of [`README.md`](README.md) under the ~250-line rule (see
[`../README.md`](../README.md)). This is the data half of the training path:
the hop that fetches the text, the shape of a packed corpus on disk, and what
phase 1 measures about both.

## Where the text comes from

`llm/corpus/remote.py` is the client side of the one hop that runs toward the
backend. The engine owns no corpora: what a run trains on is a dataset in the
*data* engine's database, and the two Python services never speak, so a run asks
the backend and the backend re-mints toward the other service carrying the same
acting caller. The seam itself is documented in
[`../backend/README.md`](../backend/README.md).

| Variable | Default | Meaning |
|---|---|---|
| `BACKEND_URL` | `http://127.0.0.1:8080` | |
| `BACKEND_SERVICE_SECRET` | unset | Shared with the backend. Unset signs the token anyway, because `act` still has to travel |

Three properties, and each of them is a decision rather than an implementation
detail:

- **Nothing is written down.** A page is fetched, encoded and dropped. A cache or
  a staging directory would be the engine owning a corpus again.
- **Pages, not the export stream.** The data engine can serve a whole dataset as
  one NDJSON response; a connection dropped forty minutes into a pack would then
  cost the whole pack, where a page costs one retry.
- **A page is sized in bytes.** `limit` counts documents and a thousand books is
  not a thousand snippets, so the last page's measured weight sets the next
  one's limit. The data engine's own ceiling of 200 rows is what binds for small
  documents; the byte target is what binds for large ones.

All three build paths read through it: the pretrain pack, the vocabulary's
strided sample, and the SFT render. Nothing in `llm/` opens a corpus file any
more.

**A selection is a dataset id, not a name.** `{"id": ..., "count": N}` caps to
the first N *documents* by `ordinal` — where it used to mean the first N files
sorted by name, which was reproducible only by accident of how a listing sorted.
A name is carried for readability and for the span tag; the data engine's own
name wins.

**There is no "everything".** An empty selection used to mean every corpus under
`input/`, which is reasonable for a directory one person curated. A shared
database anyone can pull into is not that, and a run that silently widened to
whatever was added last week would key apart every time somebody else worked. A
run names its datasets or it is refused.

**A corpus key includes what answered, not only what was asked.** A dataset
pulled again holds more documents under the same id, so the key carries the
document count and `updated_at` alongside the id and the cap. Without them a run
reuses a corpus packed from less data and reports success. A dataset's *name* is
deliberately not in the digest — but it is in the readable slug, and the slug is
part of the path, so a rename does re-pack. That is the price of a directory
listing a person can read, paid knowingly.

**A conversation arrives as `[{role, content}]`** and is rendered from
`{system?, turns: [{role, text}]}`. `sft._conversation` bridges the two, and the
part that matters is that a **system message is not a turn**: left in the list it
reaches `_turns` as an unknown role and drops the whole conversation — every
conversation, for a dataset that has them.

**A `tool` turn arrives in one of two shapes, and they must not be confused.**
The same two `ChatTurn` accepts, for the same reason:

- **It names its call** (`tool` + `argument`), and `content` is the answer's
  body alone. `_turns` renders it with `chat.tools.render_result` — the single
  renderer for the live path and the SFT data both.
- **It names no call**, and `content` is the whole rendered block already. It
  passes through untouched.

Rendering the second shape again nests one `<|tool-call|>` block inside another;
leaving the first unrendered trains the model on a result whose call it cannot
see. Neither is visible until a fine-tuned model meets the real thing, which is
why `tests/test_sft_tool_turns.py` pins the rendered turn against
`orchestrate.chat._history` rather than against a literal.

Every converted corpus in the data engine arrives in the second shape — all
138,110 tool turns of `hermes-agent-reasoning-traces-clean` are pre-rendered by
`processors/hermes.py`, which can call neither into Rust nor into this package.
The first shape is what a source with a *live* renderer behind it can use: the
observation body comes from `tools::observation` in Rust, and the call header
from `render_result` here, so neither is copied. An unknown tool drops the
conversation rather than raising — the chat path can tell a caller it was wrong,
a build over hundreds of thousands of rows cannot.

## What a packed corpus is made of

A build directory holds the tokens (`corpus.bin`, plus `mask.bin` for SFT), the
**document table** (`spans.npy`), and a sidecar (`segments.json`) recording what
the build was made from. The three are keyed together — see `corpus_key` and
`dataset_key` for what a key is a function of.

The table is separate from the sidecar, and columnar, because it is the one part
that scales with the *number* of documents rather than with their size: 22 bytes
a row against 145 for an object, and no JSON encoding of the whole table held in
memory to write it. That is affordable at one span per shard and is not at one
span per document, which is the granularity a corpus fetched from the data
engine has. `llm/corpus/spans.py` carries the measurements.

A build packed before the table moved out of `segments.json` carries its rows
inline there. Those are read, not invalidated — a repacked corpus is hours of
work, and `spans.from_sidecar` takes either shape.

## What phase 1 measures, and why it is measured where it is

`train.Tokenize` is the phase-1 frame, and its field names are the frontend's
`TokenizeStats` verbatim — a rename anywhere between here and the panel arrives
as `undefined`. A field a phase does not measure is **omitted, not zeroed**, so
"—" means nobody measured rather than "none".

Phase 1 publishes those frames **while it runs**, not only at the end of it —
`pack.pack` and `vocab.build` take a `BuildReport` from the orchestrator and say
where they are on a widening interval. What that costs, why the reporter is
passed down rather than the frame type reached for, and what each step reports
are in [`telemetry.md`](telemetry.md). Two consequences belong here, because they
are properties of the pack rather than of the reporting:

- **The live counters and the sidecar's are the same counters.** `pack._Tally`
  keys characters read and tokens written by `(stage, dataset)` at the one moment
  the text and its encoding are both in hand; the frame's `sources` and
  `segments.json`'s `fertility` are two readings of it, the second summed back
  down to the dataset name. Nothing is counted twice and nothing can disagree.
- **The pack is told what it is packing.** `build_dataset` hands it a `plan` —
  `[{dataset, stage, expected}]` in reading order, off the resolved selection —
  because the one thing a stream of `SourceDoc` cannot say about itself is how
  much of it is left. Without a plan the pack still reports what it has done and
  simply cannot project an ETA, which is what a script or a test gets.

Beyond the counters, a pretrain build reports what the tokenization and the
packing actually produced: `fertility` (chars per token, per source),
`dead_vocab` (ids the corpus never uses — tied embeddings make each one a row
paid for in every forward pass), `merge_sequence` (the head of the merge list in
learned order), `boundary_straddle` (the share of `ctx_window` windows holding an
`<|endoftext|>`) and `doc_length_hist`/`docs_over_ctx` (document lengths against
the window). Three constraints shape where each is computed:

- **Nothing gets a second pass over the corpus.** Corpora reach 4.17B tokens.
  Fertility is two counters per dataset in the pack's existing stream; dead vocab
  and the merge counts read the histogram the pack already records
  (`diagnostics.dead_ids`); the straddle and the length histogram are
  `spans.window_fit` over the document table, blocked at `MEASURE_BLOCK` rows so
  peak memory is ~42 MB whatever the build weighs. Measured at the 5B-token,
  8.3M-document scale: 0.25 s for `window_fit`, 1.3 s for the per-document
  counters spread across the whole pack, and milliseconds for the rest.
- **A metric that only a fresh build could report would read as missing on every
  reused one**, which is the same lie as a zero. So the per-dataset counters go
  into `segments.json` beside the histogram, and `train._phase_one` builds the
  closing frame off the corpus on disk for a fresh build and a reused one alike.
- **`ctx_window` is not part of the corpus key**, so nothing measured against it
  belongs in the sidecar — the same build is legitimately trained at two window
  sizes. `window_fit` therefore runs at report time, not at pack time.

The merge sequence's `count` is how often the merged token occurs in the packed
corpus, **not** the pair frequency that earned the merge: the `tokenizers`
trainer keeps that only while it is choosing and exposes neither it nor a
callback, so having it would mean reimplementing BPE training beside the library
that does it. Read the sequence's *order* as the tokenizer's story and the count
as what the corpus made of it.
