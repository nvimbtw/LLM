# The corpus a run trains on — where it comes from, what it is made of

Split out of [`README.md`](README.md) under the ~250-line rule (see
[`../README.md`](../README.md)). This is the data half of the training path:
the hop that fetches the text, the shape of a packed corpus on disk, and what
phase 1 measures about both.

## Where the text comes from

`llm/corpus/remote.py` is the client side of the one hop that runs toward the
backend. The LLM engine owns no corpora: what a run trains on is a dataset in the
data engine's database, and the Python services never speak, so a run asks
the backend and the backend re-mints toward the other service carrying the same
acting caller. The seam itself is documented in
[`../backend/README.md`](../backend/README.md).

| Variable | Default | Meaning |
|---|---|---|
| `BACKEND_URL` | `http://127.0.0.1:8080` | |
| `BACKEND_SERVICE_SECRET` | unset | Shared with the backend. Unset signs the token anyway, because `act` still has to travel |

Four properties, and each of them is a decision rather than an implementation
detail:

- **Nothing is written down.** A page is fetched, encoded and dropped. A cache or
  a staging directory would be the LLM engine owning a corpus again.
- **Pages, not the export stream.** The data engine can serve a whole dataset as
  one NDJSON response; a connection dropped forty minutes into a pack would then
  cost the whole pack, where a page costs one retry.
- **A page is sized in bytes.** `limit` counts documents and a thousand books is
  not a thousand snippets, so the last page's measured weight sets the next
  one's limit. The data engine's own ceiling of 200 rows is what binds for small
  documents; the byte target is what binds for large ones.
- **Several pages are in flight, and how many is a function of what one weighs.**
  The waiting is nearly all of a page — an offset scan, a JSON encode, two hops —
  so pages asked for together arrive in about the time one of them would.
  But a reader running further ahead than the encoder can consume is not reading,
  it is four more threads competing for the six cores the encoder is on. So
  `remote._pages_at_once` asks for however many pages make one byte target's
  worth: **one** for a corpus of books, where a single page already reaches it,
  and up to `PAGE_CONCURRENCY` for a corpus of snippets, where the 200-row
  ceiling binds first and a page is half a megabyte. Measured, packing end to
  end: 6,000 `cosmopedia` documents in 8.1 s one page at a time and 3.4 s six at
  a time; 150 `gutenberg_eng-clean` books in 7.3 s at one and 8.9 s at a fixed
  four. Sizing off the page's weight lands on the right one without being told
  which corpus it is looking at.

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

## Project and memory source roles

SFT also accepts input-only `project` and `memory` roles in the existing
`messages` list. Their `content` is a JSON string: project payloads carry
`project_id`, `name`, `description`, `notes`; memory payloads carry `memory`,
`thread`, `message`, `role` (`user` or `assistant`) and `age_seconds`.
They use the same validated payloads and canonical renderer as live chat,
without training any context token as an assistant target. Each example allows
one project and eight memories, and still ends on an assistant reply. Invalid
context JSON drops the row. Full field bounds, examples, escaping and reserved
slot adoption are documented in [project-memory.md](project-memory.md).

Packed SFT keys/sidecars record `context_format: 1`. This intentionally rebuilds
older packs that skipped the new roles; ordinary source rendering is unchanged.

`sft.build` also accepts a caller-owned `directory` for private supervised
preparation, without changing `DATA_DIR` or sharing files between vocabularies.
The storage map resolves the same four artifact names in that directory. The
existing default cache path and legacy callers retain their behavior.

An explicit `PackBudget` limits combined uint16 token plus uint8 mask bytes
(three bytes per token, including separators) and document rows before append.
Bounded mode also refuses a source conversation above 8,388,608 characters or
the policy's smaller one-character-per-admitted-output-byte ceiling, before
tokenization. Cooperative cancellation is checked per source row, encoded
example and before publication. Budget refusal, source failure and cancellation
remove both partial files, preserving earlier complete token/mask files before
publication starts. This is private preparation support; it does not establish
durable cross-file cache transactions or the SFT worker/package integration.

When a legacy vocabulary still names the memory slots as placeholders, rendering
privately adopts their names through the canonical template. The report now
records actual IDs 47/48 from those examples even though the original tokenizer
does not name them yet. They remain input-only. The SFT tokenizer-byte helper can
then retain a derived identity while leaving the base bytes unchanged.

## Builtin example vocabulary

The builtin architecture catalog also publishes a real byte-alphabet tokenizer
for its concrete numerical acceptance examples. `corpus/baseline.py` constructs
the engine's reserved block plus every byte with zero learned merges, without
fetching or fitting text. Its implicit index and complete artifact are separate
from ordinary dataset-derived vocabularies. It never replaces a pending training
tokenizer; see [builtin lifecycle](architecture-builtins.md).

## What the encoder is given, and why the pack is not slower than it is

Packing is where a rebuild spends nearly all of its time, so the shape of what
reaches the encoder is load-bearing. A batch is `ENCODE_BATCH_CHARS` of text cut
into `bpe.CHUNK_CHARS` pieces — **not** a handful of whole documents, which is
what it used to be. Three things follow, and the first two are what make an
11 GiB machine enough:

- **A batch is bounded by characters.** Filled with whole documents it was the
  target *plus the last document*, and `gutenberg_eng-clean` holds books from a
  few KB to 27.6 MB. One of those cost 30.6 s and 4.5 GiB of peak RSS inside the
  encoder; in pieces it is 5.9 s and 1.2 GiB. Packing a 60 MB slice peaked at
  1,127 MiB before and 483 MiB after.
- **Every core has work to the end of the batch.** The library parallelizes
  across a batch's *items*, so nine books are nine work units for six cores and
  the largest book is the batch. Pieces are hundreds of units and divide evenly.
- **The reader runs on a thread of its own** (`pack._read_ahead`), so a page is
  fetched while the page before it is encoded. Before that, the six encoding
  threads were busy 37% of the wall clock and the thread doing everything else
  spent 83% of its own blocked on a socket.

Together, on a 60 MB slice of the corpus above: **21.8 s to 10.9 s**, and the
`corpus.bin`, `spans.npy` and `segments.json` that come out are byte-identical
to what the old path wrote. They have to be, and two things are why.
`bpe.encode_batch` asks for `encode_batch_fast`, which is the same ids without
the token strings and character offsets nothing here reads (1.8x the CPU and
1.2x the memory for four fields nobody asks for). And a piece boundary is a
place the byte-level pretokenizer would have split anyway — a whitespace
character whose predecessor is not whitespace — so BPE, which never merges
across a pretoken, produces the same ids either way.
`tests/test_encode_chunking.py` is what holds that: the cut rule, the identity,
and a corpus packed at two different batch sizes coming out the same.

A document with no such position in it — a megabyte of base64 — is left whole.
It keeps the cost it already had, which is the only honest thing to do with it.

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
LLM engine has. `llm/corpus/spans.py` carries the measurements.

A build packed before the table moved out of `segments.json` carries its rows
inline there. Those are read, not invalidated — a repacked corpus is hours of
work, and `spans.from_sidecar` takes either shape.

Recipe pretraining preparation supplies `pack.PackBudget`: at most 16 GiB of
uint16 token bytes, including separators, and 16,777,216 document rows. Each
piece is admitted before adding token bytes or a completed row, so crossing a
ceiling cannot publish an oversized build. The temporary token file is removed
on budget/source failure; a previous complete build remains intact. Exact-fit
budgets preserve the existing token, span and sidecar bytes, including empty
documents and split encoding batches. Reused builds are checked against the
same ceilings. These are output bounds, not a tokenizer allocation estimate;
parent admission/staging capacity and worker resident-memory planning remain
separate parts of the supervised training integration.

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
  `[{dataset, stage, expected, expected_chars}]` in reading order, off the
  resolved selection — because the one thing a stream of `SourceDoc` cannot say
  about itself is how much of it is left. Without a plan the pack still reports
  what it has done and simply cannot project an ETA, which is what a script or a
  test gets.
- **The ETA is projected from text, not from document counts.** A document is
  not an amount of text: `gutenberg_eng-clean` is 86% of a typical selection's
  bytes and 5.6% of its documents. A pack a quarter of the way through the books
  had read 3% of the documents it was asked for, divided the time it had spent
  by that, and reported **fourteen hours** on a build that takes under one —
  then would have collapsed the moment it reached the snippets. So
  `pack._Tally` paces on characters against `remote.Selected.weight`, the
  dataset's own recorded `byte_size` scaled by the cap. Bytes are within 1.5% of
  characters on every corpus here, which is far inside what an ETA is worth, and
  a plan that states no text is still paced by the documents it does state.

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
