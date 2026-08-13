# Three benchmarks, not one

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
