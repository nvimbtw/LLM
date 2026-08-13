# The GPU queue

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
