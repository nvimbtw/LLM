# CLAUDE.md

> **Role:** See [`AGENTS.md`](../AGENTS.md) at the project root for the agent
> role system. Unless told otherwise, you are the **Fixer** — the only role
> that edits application code. Adhere to the coding rules in
> [`.claude/skills/rules/SKILL.md`](../.claude/skills/rules/SKILL.md).

Guidance for Claude Code when working in this repository.

## Project overview

Backend API for an LLM project (crawler, data pipeline, tokenizer, and model training/inference, fronted by a chat-style web UI). Rust backend, Next.js/TypeScript frontend. This file covers the `backend/` half.

## Tech stack

- **Web framework**: Actix-web. Not Axum — deliberately avoided for its extractor/trait-bound complexity.
- **Database**: PostgreSQL via `sqlx`. Not an ORM (no Diesel, no SeaORM) — raw SQL with compile-time query checking via `query_as!`/`query!`.
- **Auth**: JWT via `jsonwebtoken` (rust_crypto backend), passwords hashed with `argon2`.
- **Migrations**: `sqlx::migrate!`, embedded at compile time, run automatically on startup from `db::create_pool`. Migration files live in `backend/crates/db/migrations/`. The macro resolves its argument against `CARGO_MANIFEST_DIR`, so the directory and the `"./migrations"` literal must move together — mismatching them still compiles and fails at runtime.

## Directory structure

```
backend/
├── Cargo.toml         workspace root AND the binary package
├── src/main.rs        the only file here: HttpServer setup, CORS, app_data
└── crates/
    ├── storage/       on-disk layout under data/ — no dependencies at all
    ├── crypto/        argon2 hashing, AES token cipher
    ├── tokenize/      BPE tokenizer
    ├── llm/           model, CUDA kernels, training/eval          → tokenize
    ├── training/      dataset building + entry points into llm    → llm, tokenize, storage
    ├── synth/         synthetic SFT/chat generation (HTTP)        → llm, tokenize, storage
    ├── labeling/      autolabel + gather → raw/preference/        → llm, tokenize, storage
    ├── db/            one module per entity, sqlx + migrations/   → crypto
    ├── jobs/          queue + train_worker + cycle (owns the pool) → db, training, synth, labeling
    └── api/           routes + extractors + ws + jwt              → jobs, db, crypto, …
```

**Dependency order, strictly one-directional:**

```
storage · crypto · tokenize  →  llm  →  training · synth · labeling  →  db  →  jobs  →  api  →  backend
```

Nothing may depend on a crate to its right. Cargo enforces this for free — a
cycle simply fails to compile — so if a new edge does not build, the layering is
wrong, not Cargo.

`crates/*` are workspace members declared in `backend/Cargo.toml`. They are
subordinate to `backend`, not independent top-level projects — **never create a
new crate outside `backend/crates/`**, and never add a dependency to
`backend/Cargo.toml` that belongs to a crate. The binary depends on exactly
three crates (`api`, `db`, `jobs`); if that list grows, logic has leaked back
into `main.rs`.

Two boundaries that are not obvious and are load-bearing:

- **`jobs` is three modules, not three crates.** `train_worker` reads
  `queue::{ProgressTick, emit, frame, lock}` and `queue` calls
  `train_worker::{run, snapshot_frame}` and matches its `Outcome`. That is a
  cycle; separating them means first extracting the shared progress types.
- **`jwt` lives in `api`, not `crypto`.** Its `Claims` carry a
  `db::user::AccessLevel`, and `db` depends on `crypto` for password hashing, so
  `crypto::jwt` would close a loop. `api` is the only layer that issues or
  verifies a token, so it belongs there regardless.

The crate is named `api` rather than `web` because `web` collides with
`actix_web::web`, which every route file imports.

## Layering rules (strict)

- **`api/routes/*.rs`**: HTTP handlers only. Parses requests, calls `db::` functions, maps results to `HttpResponse` with correct status codes. Each file exposes a `pub fn configure(cfg: &mut web::ServiceConfig)`, registered in `routes/mod.rs` via `web::scope(...)` and re-exported as `api::configure`.
- **`db/*.rs`**: Pure SQL queries via `sqlx`. Functions return `Result<T, sqlx::Error>` or `Result<Option<T>, sqlx::Error>` — never return `HttpResponse` or `impl Responder` from this layer. Db structs (e.g. `User`) may carry sensitive fields (password hash) — never serialize a db struct directly back to a client; use a separate response struct. This rule now spans a crate boundary, so `pub` here means "visible to the whole workspace".
- **`crypto/*.rs`**: Self-contained crypto helpers (hashing, symmetric cipher). No DB or HTTP awareness — and this is now enforced rather than asserted, since the crate's manifest names neither `sqlx` nor `actix-web`.
- Composite operations that span multiple concerns (e.g. login: lookup user + verify password + issue token) live in `api/routes/`, not `db/`, since they aren't pure SQL.

## Database conventions

- Enums (e.g. `AccessLevel`, `document_status`, `job_type`) are real Postgres `ENUM` types, mapped via `#[derive(sqlx::Type)]` with `#[sqlx(type_name = "...", rename_all = "lowercase")]` on the Rust side. Don't use plain `TEXT` columns for fixed-value fields.
- Every mutable entity should have `created_at` (and `updated_at` where relevant) — required for ordering and for sidebar/recency UI on the frontend.
- IDs are UUIDs (`gen_random_uuid()` default in Postgres, `Uuid` type in Rust via the `uuid` crate).
- Migrations are additive and forward-only once data exists. Only edit an existing migration file directly while the local dev DB is still disposable/empty.

## Data storage (crawler/pipeline)

Actual data (scraped HTML, processed text, tokenized output, model weights) lives on the filesystem under `backend/data/scraped/`, `backend/data/processed/`, `backend/data/tokenized/`, `backend/data/models/` — not in Postgres — needed for manual inspection during crawling/debugging. Postgres only stores tracking metadata:
- `documents` table: one row per crawled page, tracks status, file paths, crawl depth/parent, content hash for dedup.
- `jobs` table: the GPU queue (see below).

## The GPU queue

There is one GPU, so there is one runner and one line. **Every** use of the device —
a training run, a benchmark, a chat turn — is a row in `jobs`, and `queue.rs` is the
only thing that starts one. Before this, the "one job at a time" rule was enforced by
route handlers asking two in-memory registries whether they were busy: true only for
the paths that remembered to ask (chat never did), and forgotten across a restart.

- **`jobs` owns scheduling; the domain tables own results.** A row carries status,
  `queue_rank`, coarse `progress`/`stage`, the `config` it was queued with, and a
  nullable `ref_id` naming what it produced — `models.id_model`, `benchmark_runs.id_run`,
  or `messages.id_message`. Not a foreign key: it points into three different tables.
- **`queue.rs` is an async task owning the pool**, so every durable write happens
  there: claiming (`FOR UPDATE SKIP LOCKED`), throttled progress, the terminal row,
  and each kind's domain record. Workers are blocking CUDA on plain OS threads and
  never touch the pool; they report through a `broadcast` of live frames and an
  `mpsc` of coarse progress ticks.
- **Live state is in memory and only while a job runs** (`Queue::live`): the cancel
  flag, the frame broadcast, the accumulated snapshot. Nothing in flight survives a
  restart, so there is nothing to persist about it — what survives is the queue.
  High-frequency frames (loss series, per-layer diagnostics, tokens) must **never**
  be written per-frame to Postgres.
- **`queue_rank` is a float.** A new job goes in at `max+1` over the live band; a
  drag writes the midpoint of its new neighbours, so reordering is one row rather
  than renumbering the tail. `db::job::reorder` renormalizes when bisection runs out
  of f64 precision.
- **On startup** `db::job::fail_orphaned` closes out rows still marked `running`
  (their process died); `queued` rows are deliberately untouched and the runner
  drains them, honouring the single-row `queue_state.paused`.
- **Cancellation** is cooperative at a step boundary. `llm::train`, `llm::eval::run`
  and `generate_chat_response_streaming` all take `Option<Arc<AtomicBool>>` /
  `Option<&AtomicBool>`; `eval::run` returns `Ok(None)` when it stops early, which is
  a distinct outcome from `Err`.

`train_worker.rs` holds the training-specific half (the two-phase run and its
telemetry) — it was `jobs.rs` before scheduling moved out of it.

## Two benchmarks, not one

`llm::eval` exposes `run_language` and `run_markup`. They are separate jobs, separate
`benchmark_runs` rows (discriminated by the `suite` column) and separate 0–100 scores
that **share a range and nothing else** — never compare across them. The split exists
because a supervised fine-tune is expected to move them in opposite directions, and
one blended number could not say so.

- **Language** — held-out bits per byte (0.45, through a fixed logistic), minimal-pair
  grammar (0.40), real-word rate (0.15), and distinct-3 as a *multiplicative gate*
  rather than a weighted term.
- **Markup** — marker minimal pairs (0.40), forced-position marker prediction (0.30),
  generated structure (0.20), and restraint against spraying markers into prose
  (0.10). Only the first needs no data on disk.

**Anchoring the corpus term to the n-gram floor failed twice, in both directions.**
`advantage` clamped to 0.0 on the old TinyStories corpus (nothing beat the floor);
`floor_progress` then clamped to 1.0 on FineWeb (everything beats it), pinning a
quarter of the score to a constant and hiding a 1.87 → 1.53 bits-per-byte gain. With
`distinct_3` also stuck at ~0.99 for any non-degenerate model, 40% of the old score
could not move, grammar carried 94% of what did, and a fine-tune that improved on
three components out of four scored *below* its base because it lost 2 grammar pairs
in 545. The corpus term now reads `bits_per_byte` through fixed absolute anchors
(`BPB_MID`/`BPB_SCALE`) that never clamp; the floor stays in the report as a verdict.

- **`SCORE_VERSION` / `suite_version` are load-bearing.** Any change to a scoring
  constant, the pair suite, the prompt set or the corpus split moves the number
  without the model having changed. Bump it, and do not plot or compare across
  versions — `jobs::cycle` reads `latest_score_for_model`, which is why that function
  takes a suite and why the cycle passes `Language` explicitly.
- **`Report::breakdown.score_stderr`** is the score's own sampling error (binomial,
  from the pair and word counts). It is the number a cycle acceptance threshold
  should be derived from; `min_gain` is still a guessed constant and is the next
  thing to fix.
- **`eval::score` is still the only forward path.** `token_logprobs` and
  `token_predictions` are both built on one private `for_each_position` driver, so the
  window arithmetic has one interpretation. Anything indexing a mask against either
  must check `TokenScores::aligned_with` first — dropped non-finite positions shift
  every index after them.
- **`corpus::load` is stratified.** Each file supplies an equal share of the held-out
  slice. It used to fill greedily from the sorted list, which drew all 128 KiB from
  the tails of the first two of eight files.

## Corpus ingestion (streaming)

`build_dataset` streams. Peak memory is **constant in corpus size**, measured at
~1.1 GB whether the corpus is 10 MB or 150 MB, because the two phases are separated:

1. **Learn the merges from a bounded sample.** `plan_bpe_sample` takes
   `BPE_SAMPLE_BYTES` (16 MiB) stratified **equally per dataset**, not
   proportionally — proportional would give a large web-text dataset ~99% of the
   budget and leave an early-curriculum dataset tokenizing at near byte level.
   `bpe::train`'s peak runs ~50-70x what it is handed (its pair index is ~16 B per
   input byte and is never compacted), so this constant is the memory lever.
2. **Stream-encode everything** through one reused `bpe::Encoder`, in ~4 MiB chunks,
   into a `bpe::CorpusWriter`.

**Chunks cut only on newlines, and that is exact, not approximate.**
`replace_special_characters` maps `'\n'` to `'Ċ'` and `split_punctuation` splits on
it, so a newline is already a hard pretoken boundary — chunked encoding is
byte-identical to encoding a file whole. `bpe::tests::test_chunking_at_newlines_is_seam_free`
is the guard; if it ever fails the design is invalid. A file with no newline inside
`CHUNK_MAX_BYTES` has no safe cut point and is refused by name.

Builds are cancellable between chunks and finalize atomically: corpus, merges and
segments are written to `*.partial` and renamed, so a cancelled or failed build
leaves the previous dataset untouched.

**Never train on `bpe::train`'s returned corpus.** It applies each merge corpus-wide
before the next, so it produces a segmentation `Encoder` cannot reproduce — ~33%
fewer tokens on a repetitive fixture, containing ids inference can never emit. The
old pipeline saved that corpus while inference used `Encoder`, so models were
trained and served on different tokenizations. `build_dataset` now discards it and
encodes with `Encoder`, which is why token counts differ from pre-streaming builds.

## Vocabulary layout

Ids are laid out as `[0,256)` raw bytes, then a **fixed-size reserved block** of
`pretokenize::SPECIAL_BLOCK` (64) slots for the protected markers, then BPE
merges from `bpe::MERGE_BASE` upward. Only the first `NUM_SPECIAL` slots are
allocated; the rest are padding, and `special_char_from_id` returns `None` for
them.

The block is padded rather than sized to the marker count because **merge ids are
absolute and are baked into every saved `merges.bin`**. Keying `MERGE_BASE` to
`NUM_SPECIAL` meant adding one marker shifted every learned merge id, silently
reinterpreting every tokenizer and model already on disk (an old `merges.bin`'s
first merges would decode as the newly added markers). With the padding, a new
marker takes the next free slot and nothing below it moves.

Adding a marker to `SPECIAL_TOKENS` is therefore now safe for existing models.
Raising `SPECIAL_BLOCK` is **not** — it moves `MERGE_BASE` and needs a fresh
tokenizer and a re-pretrain. A marker count past the block fails the build via a
`const` assert rather than corrupting a corpus.

**Corpus files are `u16`.** `corpus.bin` and its SFT/KTO siblings carry an 8-byte
header (`LLMC`, version, element width) followed by little-endian `u16` ids, and
the corpus stays `u16` in memory through the training loop — halving both the file
and the resident `Vec` on a machine where that is the binding constraint. Ids are
widened to `u32` one chunk at a time at the step boundary, so `train_step` and the
CUDA path are unchanged. `max_vocab` is clamped to `bpe::MAX_VOCAB` (65,536), and
`CorpusWriter` rejects an out-of-range id rather than truncating it. A corpus
written before the header fails to load with a message telling you to rebuild.

## Training modes (pretrain vs SFT)

A run has a `mode` (`TrainConfig.mode` in `training.rs`): **pretrain** (default) or
**sft** (supervised fine-tuning). Both are two-phase — build a dataset, then train
on the GPU — and share the job/WebSocket progress plumbing (`jobs.rs`); they differ
only in phase 1 and which `llm::train` entry point phase 2 drives. The route
(`routes/training.rs::start_training`) branches on `mode`.

- **Pretrain** — `build_dataset` (trains a fresh BPE tokenizer over the uploaded
  books + synthetic docs → `datasets/corpus.bin` + `artifacts/tokenizer/merges.bin`)
  then `llm::train::train` (fresh random weights). Loss is computed over every
  position.
- **SFT** — fine-tunes an *existing* base model on chat conversations, training the
  loss **only on the assistant's reply tokens**. The request carries a
  `base_model_id` (validated present + resolvable, else `400`). Phase 1 is
  `build_sft_dataset(base_model_dir)`; phase 2 is `llm::train::sft_train`, which
  loads the base weights and drives the masked step. SFT reuses the base's
  tokenizer — it never retrains BPE (that would renumber the vocab and make the
  fine-tune gibberish) — and copies `merges.bin` next to the fine-tuned weights so
  the output model loads standalone. SFT typically wants a lower `lr` and fewer
  `epochs` than pretraining.

**The masking convention** (shared by the whole SFT path, from
`chat_template::build_training_example` through the CUDA loss kernels): predicting
token `i+1` from position `i`, `mask[i] = 1` iff token `i+1` is an assistant-reply
token — from the first token after an `<|assistant|>` marker through the closing
`<|/assistant|>` marker inclusive — and `0` everywhere else (system, user,
scaffold, cross-example boundaries). Masked-out positions contribute no loss and no
gradient.

**The chat split.** `raw/sft/*.jsonl` is one set serving three jobs, partitioned by
`training::chat_destination` — a hash of each conversation's *content*, so the
split is idempotent and adding conversations never re-partitions the existing ones:

- **Pretrain (50%)** — rendered through `chat_template::render_conversation` into
  `datasets/chat/`, one file per conversation, **by `build_dataset` itself at the
  start of every run**. `Chat` is offered in the curriculum picker as a *virtual*
  dataset (`training::CHAT_DATASET`); no folder under `uploads/` backs it, and
  `list_datasets` counts it straight from `raw/sft/`. Put `Chat` in the **last
  stage and at least one early one** — annealing it in at the end is what shapes
  the register, but a marker the model meets *only* in the last stage is a marker
  it has ~0.15% of the corpus to learn. **Do not confine it to the tail.**
  Prose datasets (TinyStories/Wikipedia/FineWeb) carry no markup at all, so
  `Chat` + `Synthetic` are the *only* source of all 37 non-`<|endoftext|>`
  markers; a curriculum that first uses them in stage 4 leaves 37 embedding rows
  untrained and 37 output columns actively suppressed across the whole run.
  That is the loss-spike-to-20 failure — see the stage-boundary warm start below.
- **SFT (35%)** — what `build_sft_dataset` trains on. It filters `raw/sft/` itself.
- **Eval (15%)** — held back from both, and written to `eval/chat/heldout.jsonl` by
  `training::export_chat_eval_set`. `jobs::queue` calls it before every benchmark,
  beside `export_preference_data`, so the set derives itself per run like the chat
  documents do. It feeds `llm::eval::chat` and the markup suite's position,
  structure and reply-bits stages. (It went unwritten for a long time, which is why
  every report before the split carried `"chat": null`.)

**A run derives its own inputs.** The chat documents are rendered per run for the
same reason the tokenizer and the corpus are built per run: a training run is
self-contained, and derived data that has to be refreshed by hand goes stale
silently. There is deliberately no endpoint or script to "prepare" the chat data —
add conversations to `raw/sft/`, start a run, and it uses them. This is also why
they land in `datasets/` and not in `uploads/`, which holds only what a human put
there.

Changing the share constants re-partitions the set and moves conversations across
the train/eval boundary, which invalidates chat-metric comparability with older
runs. It is not a tuning knob.

**Prose replay.** `build_sft_dataset` also mixes in prose from the base model's
pretraining corpus, fully masked in, at `REPLAY_TOKENS_PER_CHAT_TOKEN` (1.0 — half
the gradient signal). Without it a fine-tune on a few hundred conversations has
nothing anchoring it to prose and trades language ability for the fine-tune's
register. Replay is **skipped** (and `replay_tokens` reports 0) when
`corpus.bin`'s tokenizer is not byte-identical to the base model's `merges.bin` —
its ids would mean something else, and training on them is silent corruption.

**Untrained marker warm-start.** `DatasetStats::special_ids_seen` records which
special ids a corpus contained; a fine-tune's stats record the union of its own
corpus and its base's, so the record composes down a pretrain → SFT → KTO chain.
`training::untrained_marker_ids` diffs that against the fine-tune corpus and hands
the result to `llm::train::warm_start_embeddings`, which resets exactly those rows
to the table mean. The llm crate owns no policy here — "never trained" is not
recoverable from a weight value, only from a record made at pretrain time. A base
with no record falls back to the chat scaffold, matching the old behaviour. Once
the chat data is seeded into pretraining this should reset **zero** rows; a
non-zero count in the log means the corpora drifted or a marker was added.

**Stage-boundary warm start (pretrain).** The same hazard exists *inside* a
single pretrain run, and `run_training_loop` handles it: walking the stages in
order it tracks which token ids have been trained on, and before each stage after
the first it reseeds the ids that stage is the first to use — the embedding row
via `warm_start_embeddings`, the output column via `warm_start_output_columns`.

An id a stage introduces is not merely unknown, it is **refuted**: it was a
target on every step before this one, always with label 0, so the softmax spent
the whole run pushing its logit down while its embedding row sat at random init.
The first chunk that contains it takes a ~20-nat loss and a gradient to match,
and because the rogue embedding feeds attention, one such token corrupts every
later position in the window — the whole chunk, not just that token. Reseeding
both ends from the table mean turns it back into an ordinary unknown token
(~ln(vocab) nats). The reset goes through the CPU (`sync_weights_from_gpu` →
edit → `upload_weights`) because both the f32 master and the f16 view need
rewriting; it runs a handful of times per run, so the round trip is free.

This is a safety net, **not** a licence to put the marker-bearing datasets only
at the end — see the chat-split note above. The Adam moments for a reseeded id
are deliberately left alone: a stale second moment damps its first few steps,
which is the direction you want.

**SFT dataset layout** (all under `data/`):
- Input: `raw/sft/*.jsonl` — one conversation per line,
  `{"system"?: string, "turns": [{"role": "user"|"assistant", "text": string}, …]}`,
  ending on an assistant turn. `training.rs::synthetic` can generate a templated
  starter set.
- Output (built by `build_sft_dataset`, consumed by `sft_train`):
  `datasets/sft_corpus.bin` (flat little-endian `u32` token ids, same format as
  `corpus.bin`) and `datasets/sft_mask.bin` (one `u8` per token, `1`=train /
  `0`=ignore) — the two are aligned 1:1 and length-checked before training.
  Alongside them, `datasets/sft_segments.json` records one `{start, len}` span
  per conversation (exclusive of the `<|endoftext|>` between them), which is what
  lets `run_sft_training` hand `sft_train` a `DataPlan` that **reshuffles the
  conversations every epoch** — the SFT counterpart of `segments.json`. SFT always
  shuffles: there is no curriculum order to preserve, and `cfg.shuffle` is a
  pretrain knob. A dataset built before the sidecar existed falls back to the
  whole corpus in built order.

## The attention backward, and why it was dead

`cuda_attention_backward_batched` was wrong from the first training commit
(`fa50006`) until it was fixed. Every model trained before that fix has a **K
projection frozen at its random initialization** in every layer, and Q/V
gradients computed from the wrong layer's activations. Weights from before the
fix are not salvageable — the attention was never trained. Re-pretrain.

Three defects, all in that one function:

1. **The forward's Q/K/V live in a single shared `attn` scratch**, which each
   later layer's forward overwrites. Backward runs after the whole forward, so
   it read layer `layers-1`'s Q/K/V no matter which layer it was differentiating.
   Only the last layer was ever right. (`P_all` and `X` are per-layer caches,
   which is why those were fine and the bug was not obvious.) The forward now
   writes Q/K/V into **per-layer `Workspace` caches** (`q_all`/`k_all`/`v_all`)
   and the backward reads those. Nothing about Q/K/V may move back into `attn`.
2. **`dQ` and `dK` were f32×f16 `GemmEx` calls.** cuBLAS requires both operands
   share a data type and rejects the combination outright, so those two GEMMs
   never ran: `dK` stayed at zero (hence the frozen K) and `dQ` kept holding
   `dout` from its use as a temp buffer, which is why `wq` got a large *wrong*
   gradient rather than none. The f16 caches fix the types as a side effect.
3. **Nothing checked a single cuBLAS status in the file.** A failed cuBLAS call
   writes nothing and returns a host-side code; training continues and the loss
   curve still goes down. `CUBLAS_CHECK` now wraps every GEMM and aborts. The
   check is free — the status is a host return value and forces no sync.

**The guard is `tests/attention_learns.rs`.** It trains on a random block
repeated twice: the second half is fully determined by the first, but *only*
through attention, and a fresh block each step makes it unmemorizable. Broken
attention plateaus at `ln(vocab)`; working attention reaches `ln(vocab)/2`
(measured: 4.17 vs 2.07 at vocab 64). An ordinary "loss decreases" gate does not
catch this — loss decreases on prose whether or not K is ever trained, which is
how this survived for the life of the project.

**Caching Q/K/V, not recomputing them.** The first fix recomputed Q/K/V in the
backward with three extra batched GEMMs, which cost ~1.4% throughput. The forward
already computes them, so they are cached per layer instead: V lands in `v_all`
straight from its GEMM (already f16, so free), and one elementwise kernel
(`qk_cache_f16_kernel`) emits the f16 `q_all`/`k_all` the backward needs while Q/K
stay f32 for the score GEMM — so no forward numerics change. Cost is
`3 × layers × ctx × d × 2` bytes (~4.7 MB at d256/L6/ctx512), and it bought the
1.4% back: 14.57 ms/step mean over 7 runs, against 14.80 recomputing and 14.60 for
the pre-fix (broken) code.

The caches are **allocated at `max_s` but addressed with runtime-`seq` strides**,
exactly like `head_p_all`. Forward and backward see the same `s` within a step, so
the packing agrees; a short final chunk leaves the allocation's tail untouched.

## Backward scratch must be sized by `max_s`, not by `d`

`ln2_dgrad` (the LN2 backward's `dx`) is an f16 buffer at `max_s * d`, matching
`b_sd[1]`, which is what LN1's backward has always written into. It used to
share `b_dd` with the `wo` gradient — an f32 `d * d` buffer — which fit only
while `s <= 2 * d`. At d 256 that is ctx 512 *exactly*, so every run up to
Shakedown 1 sat on the ceiling and nothing showed; Balanced 1 at ctx 1024 wrote
512 KB into a 256 KB allocation on every layer of every step and its weights are
not usable.

The overrun does not fault and the loss keeps descending — it lands in the
file-static `attn`/`ff` scratch, which is live inside the same layer's backward,
so it silently corrupts that layer's attention gradients. Measured cost at
d256/ctx1024: 0.32 nats of divergence by step 8 against ~1e-5 run-to-run noise,
and the loss stops descending monotonically. That reads as a bad hyperparameter,
which is exactly why it survived.

`Workspace::assert_scratch_fits` now checks every such buffer in **bytes** at
construction, and it is a hard panic rather than a `debug_assert` — a silent
out-of-bounds device write is not something to leave to a release build.
`tests/scratch_bounds.rs` is the regression guard, at `ctx = 4 * d`. **A buffer
a kernel fills with an `[s, d]` block must be allocated at `max_s`**; sharing
one between two element types is what caused this.

**The GPU scratch is global and single-threaded.** `attn`/`ff` are file-static
and the cuBLAS handle is shared, so two `Workspace`s stepping concurrently
corrupt each other. The training loop is single-threaded so this is fine in
production, but **GPU tests must run with `--test-threads=1`** or they fail
spuriously inside cuBLAS.

## Positional encoding: RoPE

Position is applied by **rotating Q and K inside attention**, not by adding a
table to the token embeddings. `rope_qk_kernel` rotates each adjacent pair
`(2i, 2i+1)` of a head's dimensions by `pos * 10000^(-2i/hd)` in the forward;
`rope_dqk_kernel` applies the inverse to `dQ`/`dK` in the backward. Because a
score is a rotated Q against a rotated K, what survives the dot product depends
on the *difference* of the two positions — relative position, with no
parameters. `Model::init` rejects an odd `head_dim`, which the adjacent pairing
requires.

This replaced a learned absolute table that was **never trained**: it had no
gradient buffer, no Adam master and was never synced back, so every model before
v4 ran on frozen Glorot noise for position. Removing it costs nothing learned and
takes `ctx * d` floats out of every model file.

- **The forward is applied in two places** and they must agree: the batched
  training path, and `cuda_attention_head`, which the *inference* path
  (`forward_seq` → `self_attention`) calls per head. They share the one kernel;
  its f16 cache outputs are nullable so inference reuses it without caching.
- **Angles are reduced into `[-pi, pi]` before the trig call.** This file is
  compiled `--use_fast_math`, which substitutes intrinsics whose argument
  reduction degrades past about one turn, and the angle reaches ~1e3 radians at
  long context and low pair index — exactly the long-range positions RoPE exists
  to encode.
- **The cached Q/K are the post-rotation ones**, because the backward needs `K'`
  and `Q'` (`dQ' = dS @ K'`, `dK' = dS^T @ Q'`), not the raw projections.
- `Model::max_seq_len` is the context window now. It used to be read off
  `positional_table.len()` at ~14 call sites; read it through `ctx_window()`.

`MODEL_VERSION` is **5** (`ROPE_VERSION` 4 and `FINAL_LN_VERSION` 5 name the two
boundaries that change the layout). A v3 file still parses — `read_model` reads
the positional table body and discards it, since the weights after it are at a
fixed offset — but logs a warning: such a model was trained under absolute
positions this code no longer applies *and* with a K projection that never left
random init, so its output is meaningless. Loading is allowed only so model
browsing and metadata keep working on old runs.

**`tests/attention_learns.rs` is the guard.** With the absolute table gone, the
copy task is a pure RoPE test — nothing else can supply position, so a broken or
no-op rotation cannot beat `ln(vocab)`. It reaches the `ln(vocab)/2` floor, and
escapes the plateau ~500-1000 steps earlier than the absolute table did.

## The final layer norm

`project_logits` reads a **normalized** hidden state, not the raw residual
stream. The blocks are Pre-LN (`x += attn(ln(x))`, `x += ff(ln(x))`), so each one
normalizes its own *input* and adds into the residual — nothing normalizes the
residual itself, and its scale grows with depth. Without a final norm the logits
grow with it, the softmax sharpens, and every wrong prediction costs
disproportionately more nats.

It was measurable on the two pretrains that predate it: over their epoch
checkpoints, the mean `ln_gamma` of the last block against layer 0 stayed flat at
1.09 for the 6-layer run and ran away to **1.53** (2.05 absolute) for the 8-layer
one, starting at the stage the corpus switched to web text. That is the last
block fighting a residual it cannot rescale, and it is what turned the ~20-nat
newly-introduced-marker spike into the ~200 seen on that run.

- **No new kernels.** `cuda_layer_norm` / `cuda_layer_norm_backward` are
  dimension-generic; the final norm reuses them with its own `[d]` gamma/beta,
  trained like any other 1-D parameter (undecayed, as AdamW convention has it).
- **Its activations are dedicated buffers** (`final_ln_input`, `final_ln_output`,
  `final_ln_dgrad`), not borrowed from `b_sd`. The pre-norm residual has to
  survive the loss stage so the backward can differentiate against it, and the
  block loop reuses that scratch. Do not fold them back in.
- **`Model::has_final_ln` is the switch**, and `ln_final_gamma` being empty is
  its whole representation. A pre-v5 model trained with no final norm, and a
  LayerNorm at gamma=1/beta=0 is *not* the identity — synthesizing one would
  silently change what every existing model outputs. So an old base keeps the
  unnormalized projection it learned against, through loading, training, and
  saving: `write_model` writes the length, so "absent" round-trips through the v5
  layout rather than being re-inferred.
- Cost is ~3 extra launches on a step that already runs dozens: measured
  +0.9 ms/step on the d256/L6/ctx512 shape (13.2 → 14.0 mean of 3), though the
  run-to-run spread is ±0.7 ms, so read it as a few percent rather than a firm
  number.

**The guards are in `optimizer.rs`**: `the_final_layer_norm_receives_a_gradient`
and `a_model_without_a_final_layer_norm_trains_without_one`. The first exists
because this is the *third* parameter added between the residual stream and the
logits, and the previous two (the absolute positional table, the K projection)
both shipped wired into the forward pass while receiving no gradient — in both
cases the loss curve still went down, which is exactly why a "loss decreases"
test does not catch it.

## nvcc failures must fail the build

`build.rs` checks the exit status of **both** `nvcc` and `ar`. It did not, and
`.status()` reports only whether a process *ran*: a CUDA compile error left the
previous `shaders.o` in place, `ar` re-archived it, the crate linked, and every
test afterwards ran against a stale kernel. That is indistinguishable from an
edit having no effect, and it silently invalidates any measurement taken after
it. If you are editing `shaders.cu` and a change appears to do nothing, check
that the object's mtime is newer than the source before believing it.

## Mixed-precision (f32 master weights)

GPU training keeps **all compute in f16** but holds an **f32 master copy of every
*trained* weight** as the source of truth (ISSUE-0052). Each trained parameter has
three parts in `GpuWorkspace`: the f32 master (`m_*`), the f16 view the matmuls
read (`d_*`), and the f32 Adam moments (`v_*` = 1st, `s_*` = 2nd). The invariant:

- **The master is canonical; the f16 `d_*` weight is a derived, disposable view.**
  `adam_kernel` accumulates the step into the f32 master and refreshes the f16 view
  from it in the same pass (fused — no extra kernel launch). `upload_weights` seeds
  the master from the full-precision f32 CPU weights (not the f16 round-trip), and
  `sync_weights_from_gpu` reads the master back on save (so save is lossless).
- Only the weights `gpu_adam_update` steps get a master: 10×layers plus the
  shared output projection and token embedding, plus the final layer norm's
  gamma/beta when the model carries one (see "The final layer norm").
  `d_positional` is never updated, so it has **no** master and stays f16-only.

Why: before this, the master lived in f16, so an Adam step below the f16 ULP near
the weight (~1e-3 at w≈1.0) rounded away entirely — updates vanished at small LR
and diverged in coarse quanta at large LR, capping the usable LR at ~1e-5. The f32
master lets a sub-ULP `Δw` accumulate, lifting the cap (pretrain default is now
1e-4; see `use-training.ts`). Cost: **14 B/param** (f16 `w` 2 B + f32 `m`/`v` 8 B +
f32 master 4 B), ~+40 MB on a 10M-param model; compute throughput is unchanged.

## Error handling

Distinguish failure types and map to correct HTTP status codes — don't collapse everything into 500:
- `400 Bad Request` — malformed/invalid client input
- `401 Unauthorized` — auth failure (use an identical generic message for "user not found" and "wrong password" — never reveal which one failed)
- `409 Conflict` — e.g. duplicate username (check `sqlx::Error::Database` + `is_unique_violation()`)
- `500 Internal Server Error` — genuine server-side faults (DB connection issues, hashing failure, etc.)

Never return raw error internals (`format!("{}", e)`) in the HTTP response body to the client — log server-side (`eprintln!`/`log::error!`/`tracing::error!`) and return a generic message instead.

## Code style

- Favor plain structs and free functions over trait-heavy abstractions. Avoid generics unless there's a concrete, current need for them (not "might be reusable later").
- Avoid Rust idioms that exist mainly to be clever — no unnecessary lifetime gymnastics, no premature trait objects, no builder patterns for structs with 2–3 fields.
- This is a deliberate reaction to past frustration with idiomatic-but-opaque Rust. When in doubt, pick the more direct, C-like option over the more "proper" Rust one.

## Working in this repo (Claude Code specifically)

- Implement small, well-scoped, obvious changes freely (a new route following existing patterns, a new migration, a straightforward db query).
- Before any architectural change — new layering pattern, new crate, restructuring existing modules, swapping a library — explain the reasoning and proposed shape first, and wait for confirmation before editing.
- When fixing a compiler error or bug, explain why it happened, not just the fix. The person working on this project is intentionally trying to deepen their own understanding of Rust, not offload it.
- Match the existing patterns in the codebase (route file shape, db file shape, error-matching style) rather than introducing a new convention for similar problems.
