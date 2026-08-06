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

**The auto-labeller's reply cap is the post-train cycle's most consequential
constant.** `autolabel::DEFAULT_MAX_TOKENS` was 96 and was not reachable from
`CycleConfig` at all. That capped generated replies at ~265 characters — the longest
reply in a 4,861-line catalogue was exactly 265, so it bound the whole distribution —
while `eval::chat` scores held-out replies averaging ~1400 characters (median 1149,
~295 tokens). RAFT therefore trained on replies **16x shorter** than the ones the
assistant suite scored, and every round taught the model to answer shorter than the
last: approved replies fell from a mean of 158 characters on the freshly fine-tuned
seed to 88 after ten rounds. That is the whole "the cycle regresses models" mystery,
and it was never a data-volume problem — the catalogue had 4,861 lines against SFT's
346 conversations.

It compounded through the judge, which is worth knowing before touching the weights:
`stopped_on_marker` is false for a reply cut off at the cap, forfeiting `W_FORMAT`
(0.30), so 67% of candidates ended mid-sentence and approval ran 97.3% at 50-100
characters against 44.1% at 200-250. Some of that falloff is a truncation artifact the
larger cap removes on its own and some is real (small models do degrade over long
generations) — **measure them apart before re-weighting the judge.**

Now `CycleConfig::max_tokens` (default 512), clamped by `reply_budget` to
`ctx * 3/4` so the reply cannot crowd out its own prompt, with `reserve` derived from
it rather than a flat `ctx/4`. Generation is the cycle's expensive stage, so this
costs real wall clock; that is the trade.

**The two post-train training stages were measured separately, and both lose ground.**
The cycle scores only its final model, so RAFT and KTO were indistinguishable until
the intermediate one was benchmarked by hand. On one round from `Balanced 1.01`:

| stage | model | assistant (v1) | bits/token |
|---|---|---|---|
| before any fine-tune | Balanced 1 | 38.55 | 3.7797 |
| after SFT | Balanced 1.01 | **42.25** | 3.6875 |
| after RAFT | r1 self | 38.43 | 3.7829 |
| after KTO | r1 pref | **32.69** | 3.9334 |

Those are `ASSISTANT_SCORE_VERSION` **1** scores, per token and under the pre-`17767d0`
tokenizer — do not compare them against anything measured since (see "A benchmark must
not be denominated in tokens"). The *relative* story below is unaffected: all four ran
on one tokenizer, so the differences between them are real.

RAFT costs −3.82 (1.4σ, inside noise), KTO −5.74 (2.1σ), combined −9.56 (3.5σ). RAFT
hands back essentially the whole SFT gain; KTO then goes ~6 points below the pre-SFT
base. **KTO is the only stage in the pipeline with no prose replay** — `SftDatasetStats`
and `RaftDatasetStats` both carry `replay_tokens`, `KtoDatasetStats` has no such field
and no replay code path — and after RAFT it is close to pure suppression, because the
desirable half already sits at high probability and its gradient saturates (which
`cycle.rs`'s ordering note describes as a feature). `CycleConfig::skip_preference`
exists to run the loop without it and get a clean score for the self-train half.

**RAFT is `run_sft` with a different dataset builder.** `TrainMode::Raft => run_sft`
in `train_worker.rs`; same `sft_corpus.bin`/`sft_mask.bin`, same `run_sft_training`,
same masked cross-entropy. Only phase 1 differs. "RAFT vs SFT" here is a data-source
distinction, not an algorithmic one — which is what makes changing RAFT's *input* an
in-cycle move rather than a stage confusion.

**The judge is sound; its frozen threshold is what stopped selecting.** The calibration
on record reports AUC **0.835** against 32 human ratings, well clear of `MIN_AUC` —
it can tell a good reply from a bad one. But the threshold was fitted against a
*pre-SFT* model's score distribution, and the current model clears it almost always:
approval runs 42.8% in round 1, 88.7% in round 2, then **92–96% for rounds 3–10**. At
95% approval rejection sampling rejects nothing and RAFT is training on every sample
it draws — self-amplification, not selection, and language decayed monotonically over
those rounds (59.19 → 55.95 → 56.57 → 55.47 → 53.76). The `recalibrate` flag was
written for exactly this ("once the model clears the old bar, that bar has nothing
left to say") but the cycle hard-codes `recalibrate: false`, for the real reason that
a bar moving with the model can never be cleared. **Both failure directions are real;
neither is currently handled.** Note this explains the multi-round decay only — round 1
runs at genuine 40–75% selection and still regressed.

**The reply-length hypothesis was tested and falsified.** Raising the cap worked (mean
generated reply 141 → 373 characters, max 1147) and moved the outcome essentially not
at all (−8.99 → −9.56 assistant points). Raising it was still correct — training on
replies truncated mid-sentence is indefensible — but length was not the binding
constraint. It did settle the judge's length bias: post-fix approval is flat at ~91%
from 150–600 characters and falls only past 600, where the model genuinely degrades.
The old 97%→44% falloff was the truncation artifact. **Do not re-weight the judge for
length.**

**`db::message::distinct_user_prompts` returns the most recent prompts, not the
alphabetically first.** `DISTINCT ON` forces its own key to lead the inner `ORDER BY`,
so the old single-statement form limited an alphabetically-ordered result: once the
history grew past the limit, a newly typed prompt could only enter by sorting before
the incumbent Nth. The prompt pool was effectively frozen on an arbitrary key, and
chatting with the models did not widen it. Callers also **over-fetch by
`PROMPT_OVERFETCH`**, because the limit lands in SQL, before
`gather::is_standalone_prompt` drops follow-ups and fragments — asking for exactly
`max_prompts` yields fewer usable ones and the shortfall is made up from the 24
hard-coded `SEED_PROMPTS` rather than from real history.

## Three benchmarks, not one

`llm::eval` exposes `run_language`, `run_markup` and `run_assistant`. They are
separate jobs, separate `benchmark_runs` rows (discriminated by the `suite` column)
and separate 0–100 scores that **share a range and nothing else** — never compare
across them, and **never average them**. The split exists because a supervised
fine-tune is expected to move them in *different* directions, and one blended number
could not say so.

- **Language** — can it model English. Held-out bits per byte (0.45, through a fixed
  logistic), minimal-pair grammar (0.40), real-word rate (0.15), and distinct-3 as a
  *multiplicative gate* rather than a weighted term.
- **Markup** — can it drive its own harness. Marker minimal pairs (0.40),
  forced-position marker prediction (0.30), generated structure (0.20), and restraint
  against spraying markers into prose (0.10). Only the first needs no data on disk.
- **Assistant** — are the replies any good. Held-out bits per assistant-reply *byte*
  through a logistic (`CHAT_BPB_MID` 1.62, `CHAT_BPB_SCALE` 0.45), which is the
  *whole* score. **The mapping inverts the unit**: bits per byte is a cost, so fewer
  bits is a higher score, and all three suites therefore agree that up is good. Needs
  `eval/chat/` on disk; without it the score is NaN rather than zero, because an
  unmeasured model is not a bad one.

**A benchmark must not be denominated in tokens, and this one was for a release.**
`ASSISTANT_SCORE_VERSION` 1 scored bits per *token*. The BPE fix (`17767d0`) took the
vocabulary from 2.13 to 3.79 bytes per token; a bigger token costs proportionally more
bits to predict, so every model's bits-per-token rose ~74% with no weight having
changed, and the score — a logistic centred at 3.50 — collapsed against its bottom
asymptote. Every model on disk read **0.79–1.02 out of 100**, all six inside one error
bar, at a `score_stderr` of 0.045. The suite is the post-train cycle's only objective,
so `jobs::cycle` was gating on a constant for that whole period.

The tell was that the language suite was *unaffected*, because it has always measured
bits per **byte** — tokenizer-invariant by construction. Version 2 makes the assistant
suite match: the byte count is the length of what the scored tokens decode to, which
is the reply text as written, identical however it is segmented. Measured cohort, each
model under its own tokenizer:

| model | bits/byte | score (v2) | score (v1) |
|---|---|---|---|
| Shakedown 1 | 1.6579 | 47.90 | 1.02 |
| Shakedown 1.01 (SFT) | 1.5572 | 53.48 | 1.02 |
| Balanced 1 | 1.6151 | 50.27 | — |
| Balanced 1.01 (SFT) | 1.5489 | 53.94 | 0.80 |
| Balanced 1 NT | 1.7221 | 44.35 | — |
| Balanced 1.01 NT (SFT) | 1.6434 | 48.70 | 0.84 |

Every fine-tune now beats its own base by 3.6–5.6 points against a ~1.2-point margin
on the difference — the property the suite exists to report and which v1 could not.
`bits_per_token` is still in the report, since it is the loss SFT actually minimizes,
but it is only comparable between models sharing a tokenizer. **Nothing recorded at
version 1 is comparable with version 2**, and the gap cannot be undone arithmetically:
a v1 score depends on the bytes-per-token of a tokenizer the report never recorded.
`tests/calibrate_assistant_anchor.rs` re-derives the anchor if the holdout changes.

**The assistant suite is the post-train cycle's objective, and the only one that is.**
The other two are things a chat fine-tune can trade away. For one release the chat
metric lived inside `Report` as a reported-but-unscored field, because the cycle
judged on `BenchmarkSuite::Language` and had no other way to see the number it was
optimizing — that workaround is gone, and `Report` no longer carries `chat` at all.

**`jobs::cycle` runs all three per round and gates on a vector test**: a round is kept
only if **no suite regressed and at least one improved**, each against its own margin.
Not a weighted sum — a sum is exactly what lets a large gain on one suite pay for a
collapse on another and still report progress, and with three numbers there are more
places for that to hide, not fewer. "A balanced high score on all three" is a
constraint, not an average. `CYCLE_SUITES` is the list, `judge` is the policy, and
both are unit-tested without a database or a GPU.

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
  versions — `jobs::cycle` reads `latest_evidence_for_model`, which is why that
  function takes both a suite *and* a version, and why `CYCLE_SUITES` pairs each suite
  with the version it is comparable at. Each suite versions independently:
  `SCORE_VERSION`, `markup::MARKUP_SCORE_VERSION`, `ASSISTANT_SCORE_VERSION`.
- **`Report::breakdown.score_stderr`** is the score's own sampling error (binomial,
  from the pair and word counts), and `jobs::cycle::judge` now derives its margins
  from it rather than from a constant. A round is measured against the standard error
  of the *difference*, `sqrt(a² + b²)` over the two runs, so the cycle cannot turn on
  movement its own benchmark cannot resolve. The old flat `min_gain` of 0.5 sat
  *below* that floor — the composite reports ±0.9 to ±1.0 per run, i.e. ±1.37 on a
  difference — which is how a 0.41σ drop (58.94 → 58.39) came to halt a cycle.
  `min_gain` survives only as a *floor* on each suite's margin: it can make the test
  blunter, never finer than the instrument. The same margin decides both directions,
  so movement too small to count as a regression is also too small to count as a gain
  — which is what stops the gate ratcheting on noise.
- **`chat::ChatMetrics::bits_per_token_stderr` clusters on the conversation, not the
  token.** Tokens inside one reply are nowhere near independent — a reply that goes
  wrong goes wrong for fifty tokens together — so a binomial over `tokens_scored`
  understates the error several-fold and the gate would read noise as progress. The
  assistant suite propagates it through the logistic's slope
  (`100·σ·(1−σ)/CHAT_SCALE`, ~41.7 points per bit at the midpoint) into
  `breakdown.score_stderr`, so all three suites hand `judge` an error bar in the same
  unit as their score. A missing error bar stays NaN and never becomes zero, which
  would make an absent measurement look like a perfectly certain one.
- **`MarkupReport` still carries `chat`, reported and unscored.** Harmless duplication
  of one cheap forward pass, kept because the markup suite's position and structure
  stages already load those conversations.
- **`eval::score` is still the only forward path.** `token_logprobs` and
  `token_predictions` are both built on one private `for_each_position` driver, so the
  window arithmetic has one interpretation. Anything indexing a mask against either
  must check `TokenScores::aligned_with` first — dropped non-finite positions shift
  every index after them.
- **`corpus::load` is stratified.** Each file supplies an equal share of the held-out
  slice. It used to fill greedily from the sorted list, which drew all 128 KiB from
  the tails of the first two of eight files.

## The marker grammar is enforced while decoding

`llm::markup::OpenBlocks` is a pushdown automaton over the reserved marker block.
`infer::stream_generation` masks every illegal marker's logit to `-inf` *before*
sampling, so a malformed reply is not merely unlikely — it is unrepresentable.

The model cannot do this on its own and is not close. The markup suite scores
`closer_matches_opener` at **52.2%** against a 50% chance baseline and
`nesting_order` at 63.3%, and served replies showed exactly that shape: `<|par|>`
closed by `<|/item|>`, blocks left hanging, and a whole `<|/user|><|assistant|>`
turn boundary invented mid-reply. The rules are the four the suite tests plus two
containment ones — `<|item|>` only directly inside `<|list|>`, and only `<|item|>`
directly inside it — with `MAX_DEPTH` 6 bounding a model that opens and never closes.

- **Ordinary text is legal in every state, unconditionally.** That is what
  guarantees masking cannot empty the candidate set, and it is asserted rather
  than assumed (`ordinary_tokens_are_always_legal`).
- **The end-of-turn markers are gated on an empty stack.** This is also what makes
  the model *close* its blocks: it cannot reach `<|/assistant|>` any other way.
- **The table fails closed.** A marker in `SPECIAL_TOKENS` with no rule in `PAIRS`
  or `STANDALONE` is forbidden, so adding one cannot silently make it emittable.
- **The token cap is the one stop the grammar cannot refuse**, so the loop reserves
  `depth() + 1` tokens against `max_tokens` and drains the open blocks when it
  stops. The `+ 1` is because the token about to be sampled may itself open a
  block; without it a reply that opens on its last token overruns the cap by one.
- **`generate_response_streaming` passes `None`** — a bare prompt completion is raw
  continuation, not an assistant turn, and has no scaffold to keep balanced.

**It must stay out of `eval`.** The markup suite exists to measure whether the model
has *learned* the grammar; generating its structure stage under the constraint would
measure the constraint and score every model alike. `eval::generate::continue_tokens`
therefore does not use it, and the suite's score is still the number to watch — this
changes what the model *emits*, not what it knows. Expect the served output to look
dramatically better while the benchmark does not move at all; that is correct, and
the gap between them is the honest measure of how much work the constraint is doing.

## Corpus ingestion (streaming)

`build_dataset` streams. Peak memory is **constant in corpus size**, measured at
~1.1 GB whether the corpus is 10 MB or 150 MB, because the two phases are separated:

1. **Learn the merges from a bounded sample.** `plan_bpe_sample` takes
   `BPE_SAMPLE_BYTES` (16 MiB) stratified **equally per dataset**, not
   proportionally — proportional would give a large web-text dataset ~99% of the
   budget and leave an early-curriculum dataset tokenizing at near byte level.
   `bpe::train` deduplicates the sample to its distinct pretokens first, so its
   peak scales with the sample's *vocabulary* rather than its length (measured:
   267 MB peak to train a full 8192 table on 10.6 MiB and then encode 8 MiB of
   held-out text). This constant is no longer the memory lever it was — see the
   note in `plan_bpe_sample` on why its **composition** is now the open question.
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

**The trainer and the encoder apply merges by the same rule, and that is the whole
correctness story of the tokenizer.** `bpe::train` applies each learned merge to
every *word* independently, in rank order, to fixed point — exactly what
`Encoder::merge_word` does when it repeatedly takes the lowest-id applicable merge.
`bpe::every_learned_merge_is_reachable_by_the_encoder` is the guard, and it is a
total check over the table rather than a sample.

It has to be, because the failure is silent and expensive. The trainer used to walk
a flat corpus while tracking pair positions in a side index, and the index went
stale the moment a merge shortened a word: every later position shifted left by one
and was never corrected, so occurrences were skipped and the merges learned after
that point were fitted to a segmentation `Encoder` cannot reproduce. On the real
8192 table that left **3,004 of 7,872 merges (38%) unreachable** — `" the"` was
learned as `" th" + "e"` at rank 6, but the encoder reaches `[" t", "he"]` and
stops, so `" the"` sat in the vocabulary and could never be emitted. The casualties
were the most frequent words in English (`" the"`, `" in"`, `" from"`, `" they"`,
`" have"`, `" were"`), 51% of emitted tokens were single characters, and
bytes-per-token sat at **2.13**. Nothing downstream noticed: the vocabulary looked
full and the models trained fine, just against a third less compression than they
should have had. Rebuilding the same 8192 table on the same data now gives **3.79
bytes/token, 17% single-character, 0 fragmenting merges**.

The invariant is "a merge's text re-encodes to a **single** token", not "to that
exact id". Those differ only for aliases — two ids decoding to the same text, of
which the encoder naturally reaches one. On a real table that is 4 merges, all a
lone space, costing 4 embedding rows and no compression.

Because the two now agree, `bpe::train`'s returned corpus *is* what `Encoder`
produces (`the_returned_corpus_matches_what_the_encoder_produces`). `build_dataset`
still discards it and stream-encodes instead, so there is exactly one code path
producing token ids.

**Any change here needs a fresh tokenizer and a re-pretrain.** Merge ids are
absolute and baked into every `merges.bin`; a model carries the vocabulary it was
trained under.

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

## The thinking budget

`chat_template::ThinkBudget` (`low`/`med`/`high`) renders as **one reserved marker
immediately before the `<|assistant|>` it governs**:

```text
<|user|>U<|/user|><|think:high|><|assistant|>
```

`synth::teach` has always written this label — it files matched triples into
`raw/sft/think-{low,med,high}.jsonl` by *measuring* the thinking block in the
student's tokenizer — but `SftConversation` declared only `system` and `turns`, so
serde dropped it and the tiers trained as unexplained variance between otherwise
identical replies. It is now a field on `SftConversation`, threaded through all
three renderers, the chat socket and the composer's dropdown.

- **`None` renders nothing, byte-for-byte.** Every corpus, `chat_destination`
  hash and benchmark score already on disk depends on this, and
  `no_budget_renders_exactly_as_before` is the guard. `conversation_hash`
  deliberately does **not** absorb the label — folding it in would re-partition a
  set that has already been trained and scored against.
- **The marker is a cue, never a target.** Masked `0` like `<|assistant|>` itself:
  it is an instruction the caller supplies, not something the model chooses.
  Training it would teach the model to predict how hard it is about to think. Only
  the embedding row is ever learned; the output column stays unused.
- **It is per-turn, and history is unmarked.** `build_prompt` strips historical
  reasoning, so marking a history turn would claim a thinking block that is not
  there. The rule holds in both directions: *marker ⟺ the assistant turn after me
  shows its reasoning at this budget*.
- **It is charged to the scaffold.** `build_prompt` adds the marker's token to the
  mandatory cost before fitting history around it, or a full context would
  overflow by exactly one token, silently, only on budgeted turns.
- **Adding the markers needed no retrain.** `SPECIAL_BLOCK` is a fixed 64 against
  44 markers, so ids 300–302 were already allocated padding rows — `MERGE_BASE`
  does not move and no `merges.bin` is invalidated. The rows are at random init,
  and the existing `untrained_marker_ids` → `warm_start_embeddings` path resets
  them (all four models on disk record `special_ids_seen`, so they take the real
  diff rather than the `LEGACY_SCAFFOLD_MARKERS` fallback).
- **`synth::teach::Tier` *is* `ThinkBudget`**, re-exported, not a twin — three
  enums that happened to serialize the same way would be three places to drift.

Generation paths (`labeling::autolabel`, `labeling::gather`) pass `None` on
purpose: what they produce becomes RAFT/preference data, which is rendered
unbudgeted, and generating under a marker the data is never labelled with would
put the two out of step.

**The tier boundaries are derived from the student's context window, and must
be.** `synth::teach::derived_bounds` takes 90% of the window (the slack
`Budget::word_capacity` already holds back), gives 45% of that to the thinking
block because the reply has to fit beside it, and splits the result at 15%/60%.
At ctx 1024 that is low <= 62, med <= 248, high above.

The previous fixed constants (140/700) made the top tier **arithmetically
impossible**, and this is the failure to remember. A record is filed `high` only
when its thinking block exceeds `med_max`, and `teach::run` separately discards
any conversation exceeding the student's window. 700 tokens of reasoning plus a
reply plus the user turn does not fit 1024 — measured blocks on the long answers
ran 982-1056 tokens, over the whole window before the reply existed. Every
high-tier candidate was generated, validated, and then silently dropped. It read
as a markup problem for two rounds of investigation because the rejection tally
is broadcast to the live feed and never persisted.

`tests::a_high_tier_record_can_actually_fit_the_student` is the guard: the
smallest possible high-tier record must fit the window at every ctx. `0` in
`low_max_tokens`/`med_max_tokens` means "derive"; an explicit value is honoured.

**Re-tiering needs no GPU.** Every record stores its own `think_tokens`, so
`scripts/retier_think_files.py --ctx N` re-files existing data under new bounds.
Run it after any boundary change — a corpus holding records labelled `low` under
two different rules teaches the marker two meanings, which is worse than either
rule alone.

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
  documents do. It feeds `llm::eval::chat` — in **both** suites, see
  `Report::chat` — and the markup suite's position and structure stages. (It went
  unwritten for a long time, which is why every report before the split carried
  `"chat": null`.)

  **The overlap between `eval/chat/heldout.jsonl` and `raw/sft/*.jsonl` is by
  design, not contamination.** The holdout is *derived from* those same files by
  `chat_destination`, so every held-out conversation is also a line in `raw/sft/`;
  what makes it held out is that `build_sft_dataset` and the pretrain chat renderer
  each filter to their own share. Comparing the two directories on disk shows 100%
  overlap and means nothing. Do not "fix" it by deleting those lines from
  `raw/sft/` — that removes the Eval share from the only files it is derived from,
  and the next `export_chat_eval_set` re-partitions what is left and writes a
  *different* holdout.

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

- **The rotation is applied in three places** and they must agree: the batched
  forward (training *and* inference prefill, which now share
  `cuda_attention_forward_batched`), and `rope_decode_kernel` for a single new
  token against the KV cache. The first two are literally the same launch of
  `rope_qk_kernel` — its f16 cache outputs are nullable, so inference reuses it
  without caching. The decode kernel is separate only because Q and the cached K
  row no longer share an index space, and it calls the same `rope_cos_sin` /
  `rope_rotate` device helpers so the arithmetic still has one definition. See
  "Inference: one device-resident session".
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

## Inference: one device-resident session

`training::infer_session::InferSession` is the forward-only counterpart of
`Workspace`, and it is now the **only** inference path. `Model::forward_seq` and
its per-stage helpers (`layer_norm`, `self_attention`, `feed_forward`, `unembed`)
are gone; so is `cuda_attention_head`, the per-head kernel only they called.

The old path held nothing on the device between calls. Every layer norm, every
attention head, every feed-forward and the whole `[d, vocab]` output projection
were flattened on the host, converted f32→f16 element by element, `cudaMalloc`'d,
uploaded, launched, and copied straight back into `Vec<Vec<f32>>` — **once per
generated token**, ~1000 launches and ~250 malloc/free pairs each time. Measured
at d256/L6/h4/ctx512 with `LLM_PROFILE_INFER=1`: 66% of a token was host
marshalling and upload, 17% device→host copies, 4% host residual loops, and
**14% was kernels**. That is what low GPU utilization during generation meant.

Every one of the four pathologies commit `6048702` fixed for training was still
present here: per-head host loops, `cudaMalloc` per call, blocking D2H, and
non-persistent weights. Weights are now uploaded once, activations never leave
the device, and the only D2H per call is the logits.

**Measured, at d256/L6/h4/ctx512 on a 3060** (`tests/bench_generate.rs`, A/B run
with both paths present before the old one was deleted):

| prompt | old tok/s | new tok/s | speedup |
|--------|-----------|-----------|---------|
| 32     | 26.0 | 443.9 | 17.1x |
| 128    | 22.5 | 382.8 | 17.0x |
| 512    | 15.4 | 439.0 | 28.5x |

- **`ensure_attn` was reallocating every token.** It grows to the exact size
  requested, and a generating sequence gets one longer each step, so the guard
  missed every time and all 11 buffers — including the six backward-only ones
  inference never reads — were freed and re-malloc'd. `cuda_infer_reserve` sizes
  it once at `ctx` in `InferSession::new`. Worth ~9 ms/token at a short prompt.
- **The old path silently disagreed with the training forward.** It accumulated
  the residual stream in host f32 where training uses `cuda_add_inplace_f16`, and
  it skipped both the post-embedding and post-residual `cuda_clamp_f16` calls.
  A model was served through a different function than it trained under.
  `tests/infer_agrees_with_training_forward.rs` now pins them together and is the
  guard on any future change to either.

### The KV cache

`cuda_attention_decode` handles one new token against the cache: four
strided-batched GEMMs plus `rope_decode_kernel` and `decode_softmax_kernel`.

- **K is cached in f32, V in f16**, because that is what the prefill's GEMMs
  natively produce (`attn.k_f32`, and V straight out of its f16 GEMM). Caching K
  as f16 would make a decode step's score GEMM lower precision than a prefill's,
  and the two have to agree.
- **A prefill fills the cache as a side effect.**
  `cuda_attention_forward_batched` gained a nullable `k_cache_f32` and a
  `kv_stride`; training passes null/0, which leaves every stride at its old
  `seq`-based value. There is no separate "warm the cache" pass.
- **Decode needs no head concat.** `[h, 1, hd]` contiguous *is* the `[1, d]` row
  the wo GEMM wants. The prefill needs `cuda_concat_heads` only because its
  `[h, s, hd]` has to be interleaved by token.
- **`cached_tokens` holds the *windowed* sequence**, and that is what makes a
  moving context window safe without tracking an offset: K and V at position `i`
  are a function of the windowed tokens `0..=i` and of `i` (through RoPE) and
  nothing else, so two windowed sequences sharing a prefix share valid cached
  rows however the window moved. A slid window fails the prefix comparison and
  re-prefills, which is correct and is `a_sliding_context_window_stays_correct`.

**A trap when writing tests against this.** Two `logits_*` calls on the *same*
session no longer run the same code: the second finds the sequence cached and
takes the decode branch. Comparing them therefore compares prefill against
decode, which is a real property but not usually the one being tested, and it is
not bit-exact (see below). Use a fresh `InferSession` per side when the property
under test is anything else — truncation, a row offset, weight packing. Two tests
had to be fixed this way after the cache landed.

**What the cache is worth depends entirely on context length,** and at this
model's size it is not the headline. `prefill_ms` in the bench is a free A/B —
before the cache every token cost a full prefill of the sequence so far:

| prompt | prefill (= old per-token) | decode ms/token | speedup |
|--------|---------------------------|-----------------|---------|
| 128  | 2.0 ms  | 1.8 ms | 1.1x |
| 512  | 2.3-3.2 | 1.7-1.9 | 1.2-1.9x |
| 2048 | 9.8-10.1 | 1.9-2.1 | ~5x |
| 4000 | 34.3-36.6 | 1.8-2.0 | **~19x** |

Decode is flat in context length; prefill is not, and must not be — it contains
the O(s²) attention.

**A decode step at these shapes is launch-bound, not arithmetic-bound**: ~100
kernel launches against ~0.1 ms of FLOPs. `layers` is the only shape dimension
that reliably moves the number, because launches scale with it; rows like
`d 256→128` measuring *slower* are the noise floor, not physics. Do not read the
per-stage profile as FLOP attribution — with `LLM_PROFILE_INFER=1` each stage is
bracketed by a `cudaDeviceSynchronize`, and at ~2 ms/token those syncs cost more
than the stage does. **Fusing the residual+clamp pairs, or CUDA graphs for the
decode step, is where the next multiple is** — not more algorithmic work.

### Two host costs that were larger than they looked

- **`bpe::decode` rebuilt its merge table on every call** — ~7,900 hash inserts
  at vocab 8,192, measured at 292 µs, i.e. 15% of a token, for a table that never
  changes. The streaming loop called it once per generated token to re-decode the
  reply so far. `bpe::Decoder` indexes once, exactly as `bpe::Encoder` already
  did for the other direction; the free `decode` is now a wrapper.
- **The sampler sorted the whole vocabulary** to keep 40 candidates: 106 µs, ~5%
  of a token. `sample::sample` partitions with `select_nth_unstable_by` instead
  (5.9x faster) and its comparator breaks ties on ascending token id so the order
  is *total* — without that, an unstable partition could reorder tied logits and a
  seeded benchmark would stop reproducing its own past runs.

There were also **two samplers that had drifted apart on correctness, not
determinism**. `infer::sample_token` exponentiated raw logits with no
max-subtraction, so a confident model overflowed `exp` to `inf` and the whole
distribution went `NaN`; and `temperature = 0.0` divided by zero, returning the
argmax only by accident (`r -= inf` gives `NaN`, `NaN <= 0.0` is false, so the
loop fell through to its initial value). Its unit test passed on that accident.
There is now one sampler, taking the RNG as a parameter — which is the only thing
serving and benchmarking actually needed to differ about.

### Scores may have moved, and `SCORE_VERSION` was deliberately not bumped

`eval::score` and `kto::reference_logprob` run on the same forward, so this work
can shift a benchmark number with no model having changed. That was a considered
decision, not an oversight.

Measured on real trained weights (Shakedown 1, d256/L6) against the *training*
forward, which is the function a model's own objective implies:

- worst position: **0.098% of the logit range**, total variation between
  next-token distributions **0.003**
- **1 argmax flip in 64 positions**, and at that position the top-2 gap was
  `0.007812` against a noise floor of `0.007812` — one f16 ULP, an exact tie
  where either answer is equally defensible

So metrics built on strict argmax or `>` comparisons (minimal pairs, top-1 at
markers) can move slightly, but only where the model was already indifferent.
For scale, the markup suite reports `± 1.3` points of sampling error on its own.
No pre-change score was available to diff against — `data/artifacts/models` holds
weights but the scores live in Postgres — so **the shift was bounded by
construction rather than measured end to end.** If a suspicious score movement
shows up on a model with history, this is the first thing to suspect.

Post-change reference, for future comparison: Shakedown 1 scores **40.9 / 100**
on the language suite (bits/byte 1.796) and **32.1 ± 1.3 / 100** on markup,
reproducible across runs.

### Lifecycle

`infer::MODEL_CACHE` holds the session, keyed on the model directory, and drops
the previous one before building the next so two models' weights are never both
resident. `infer::release()` frees it, and `jobs::queue` calls that before every
non-chat job: one GPU, one queue, and the `attn`/`ff` scratch the session sizes at
`ctx` is the same file-static arena a training `Workspace` resizes. Chat jobs are
exempt, since releasing right before generating would throw away the cache that
makes consecutive turns cheap.

`InferSession` is `Send` but deliberately **not** `Sync`. A CUDA device pointer
belongs to the process's primary context, so moving a session between threads is
sound; two threads *using* one concurrently is not, and neither is two sessions
running at once, for the file-static-scratch reason above.

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
