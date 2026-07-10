---
id: ISSUE-0030
title: "Epic: Post-training (SFT) — fine-tune a pretrained base model on chat data with loss masking"
scope: cross
priority: P2
status: todo
rule: none
location: backend/src/llm/src/chat_template.rs backend/src/llm/src/train.rs backend/src/llm/src/training/transformer.rs backend/src/training.rs frontend/components/training/config-form.tsx
blocked-by: [ISSUE-0031, ISSUE-0032, ISSUE-0033, ISSUE-0034, ISSUE-0035, ISSUE-0036, ISSUE-0037, ISSUE-0038]
blocks: []
---

## Description
**Tracking/epic issue — no code lives here directly.** It is `blocked-by` the
whole child chain (0031–0038); once those are `done`, the Builder closes this one
by running an end-to-end SFT smoke test and updating the docs (see "Definition of
done" below). It exists to give the Builder the map before it picks up any child.

### The goal
Add a **post-training (supervised fine-tuning / SFT)** path: take an already
*pretrained* base model and fine-tune it on **chat-formatted conversations**,
computing the loss **only on the assistant's reply tokens** (not on the system
prompt or the user's turns). This is the step that turns a raw next-token
predictor trained on PG19 prose into something that answers in the
system/user/assistant chat shape the inference side already speaks.

### Why now / why it fits
The inference half is already built and explicitly waiting for this. See the
module doc at the top of `backend/src/llm/src/chat_template.rs`:

> "in time the synthetic-data generator and the SFT loss-masking must render the
> exact same shape, so the format the model is *trained* on and the format it's
> *prompted* with can't silently drift apart."

So the format contract (`<|system|>…<|/system|>`, `<|user|>…<|/user|>`,
`<|assistant|>…<|/assistant|>`; special markers map to PUA chars in
`tokenize::pretokenize::SPECIAL_TOKENS`) is fixed and shared. What's missing is
purely the *training* side: a masked loss, a mask-carrying dataset, a fine-tune
loop that starts from existing weights, and the orchestration/UI to drive it.

### Current state (the gap)
- `train.rs::train` always calls `Transformer::init(...)` — fresh random weights,
  no "continue from a base model" for a *new dataset*. (`continue_train` resumes
  the **same** corpus; it is not SFT.)
- `transformer.rs::compute_loss` + the CUDA `cross_entropy_*` kernels compute loss
  over **every** position `0..s-1` and normalize by `seq_len`. There is no way to
  say "ignore these positions."
- The dataset pipeline (`backend/src/training.rs::build_dataset`) trains a **fresh
  BPE tokenizer** and emits a flat `corpus.bin` of raw prose token ids — no chat
  structure, no mask.

### The design (child issues)
| # | Issue | Layer | Depends on |
| --- | --- | --- | --- |
| 0031 | Render a conversation to `(token_ids, loss_mask)` in `chat_template` | backend/llm | — |
| 0032 | Masked cross-entropy loss + backward CUDA kernels | backend/llm | — |
| 0033 | Thread the mask through `train_step` (a masked SFT step) | backend/llm | 0032 |
| 0034 | SFT dataset builder: conversations → `sft_corpus.bin` + `sft_mask.bin` | backend | 0031 |
| 0035 | SFT training loop + `sft_train` entry point (fine-tune from a base model) | backend/llm | 0031, 0033, 0034 |
| 0036 | Synthetic chat-data generator (produce the SFT conversation set) | backend | 0031 |
| 0037 | Backend orchestration: `TrainMode` + SFT route/job wiring | backend | 0034, 0035 |
| 0038 | Frontend: fine-tune mode toggle + base-model picker | frontend | 0037 |

### The masking convention (the one idea every child shares)
Training predicts token `i+1` from position `i`. "Train only on the assistant
reply" therefore means: **`mask[i] = 1` iff the token being predicted (token
`i+1`) lies inside an assistant span** — from the first token *after* an
`<|assistant|>` marker through the `<|/assistant|>` marker **inclusive** (so the
model learns to produce the reply *and* to emit the stop marker), and `0`
everywhere else (system, user, scaffold, and cross-example boundaries). Every
child that touches tokens (0031, 0032, 0033, 0034) must honour exactly this rule
so the mask built at data-prep time lines up with the loss computed on GPU.

### Definition of done (what the Builder verifies to close THIS issue)
1. All children 0031–0038 are `done`.
2. A tiny end-to-end smoke test passes: pretrain a 1–2 layer toy model a few
   steps, then run `sft_train` on a handful of hand-written conversations, and
   assert the assistant-region loss drops while a masked (user/system) token's
   logits are provably untouched by the SFT gradient (extends the existing
   `test_overfits_single_sequence` style in `tests/train_test.rs`).
3. `backend/CLAUDE.md` (or a short `docs` note) documents the two training modes
   and the SFT dataset layout.

## Comments
### Auditor · 2026-07-10
Filed as the design/tracking parent for the post-training feature the user asked
to scope out. Deliberately structured as an epic with a `blocked-by` chain rather
than one giant issue, matching this repo's preference for small, single-location,
independently-verifiable units (cf. the R1 function-splitting issues). Children
carry all the code; this issue carries the plan and the final integration check.
Priority P2: it's net-new feature work, not a latent bug, and each child is
sized to pass a single Builder→Verifier round.

### VCS · 2026-07-10
Committed as `d5b9a57` on branch `dev`.
