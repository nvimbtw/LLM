# The thinking budget

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

**The SFT set is smol-smoltalk, and the scale changed what several things mean.**
`scripts/prepare_smoltalk.py` rebuilds `raw/sft/` from HuggingFaceTB/smol-smoltalk:
484,570 source conversations in, 454,215 out (93.7%), markdown converted to the
marker vocabulary and every conversation truncated to the longest whole-pair prefix
that fits the window. It replaced a 1,912-conversation set that was largely
self-generated. The previous set is in `raw/sft-backup-pre-smoltalk/`.

Three things broke quietly at 240x the volume, all now fixed:

- **`build_sft_dataset` had no length guard**, alone among the builders — KTO and
  RAFT have always had `skipped_too_long`. An over-long conversation became a span
  in `sft_segments.json` longer than the window `sft_train` is handed. Now skipped
  and counted; 454 of 158,361 (0.29%) on the current set.
- **Prose replay silently under-delivered 7x.** It took one pass over the pretrain
  documents at `MAX_REPLAY_DOC_TOKENS` each, which caps at ~7.1M tokens over the
  11,104 documents on disk. That was ample against a budget set by ~350 chat
  conversations and is 14% of one set by 158,000 — the fine-tune would have been
  anchored at a seventh of the intended strength with `replay_tokens` the only sign.
  It now takes further passes at advancing offsets, breadth before depth, and stops
  short rather than repeating the corpus. Measured: 7.1M → 52.4M, exactly the
  1.0-per-chat-token budget.
- **The eval share is a fraction, so it grew without bound.** 15% of 454k is ~68,000
  held-out conversations, one forward pass each: a 2.5-hour benchmark where it was
  40 seconds, three times per cycle round. `MAX_HELDOUT_CONVERSATIONS` caps it at
  1,000, selected by the same content hash the split uses so the choice is stable.

The trainable share is now **86%**, against the "well under 50%" this file used to
document. That was an observation about the old set, not an invariant: smoltalk
replies are long relative to their prompts, and half the remaining tokens are replay,
which is trainable by design. 52.4M chat-reply tokens against 52.4M prose is the
50/50 the replay ratio has always specified.

**The assistant suite's anchor was re-derived and the old fine-tunes lost their
gains.** Against the previous holdout each SFT model beat its base by 3.6–5.6 points;
against this one none does (−0.3 to +0.2, inside a 0.9-point margin). That is the
honest reading rather than a broken suite: the old holdout was drawn from the same
self-generated set those models were tuned on, so it was measuring register agreement
rather than assistant ability. `CHAT_BPB_MID` is now 1.60, the mean of the cohort on
the new holdout. Every model on disk predates the swap.

**Thinking blocks are gone from SFT.** smol-smoltalk has no reasoning traces, so
`<|thinking|>` and the three `<|think:*|>` budget markers now train on nothing —
the composer's dropdown is a lever attached to nothing until data with reasoning
exists again. `prepare_smoltalk.py --keep-thinking` re-mixes the old `think-*.jsonl`
alongside if that is worth paying for.

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
