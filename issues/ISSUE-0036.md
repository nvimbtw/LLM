---
id: ISSUE-0036
title: Synthetic chat-data generator for SFT
scope: backend
priority: P3
status: todo
rule: none
location: backend/src/training.rs backend/src/storage.rs:33-35
blocked-by: [ISSUE-0031]
blocks: []
---

## Description
### What
A generator that produces **chat-formatted conversation examples** for SFT,
written as JSONL into the `raw/sft` dir (ISSUE-0034's input format). This is the
"synthetic-data generator" the `chat_template.rs` module doc names alongside the
SFT loss-masking as the two missing training-side pieces.

### Why
SFT needs conversation data to fine-tune on, and hand-writing enough is tedious.
A small templated generator gives a repeatable starter set so the whole SFT path
can be exercised end-to-end without a data-collection project first. It is
explicitly *opt-in* and separate from the loop, so it can evolve independently.

### Shape of the fix
- Generate simple, verifiable instruction/response conversations from templates,
  e.g. arithmetic ("What is 7 + 5?" → "12"), format-echo, list/sort tasks,
  short factual Q&A from a small table — categories where the *correct* assistant
  reply is programmatically known, so the data is clean by construction.
- Emit each as a JSONL line matching ISSUE-0034's schema
  (`{"system"?, "turns":[{role,text}...]}`), ending on an assistant turn. Optional
  system prompts on some fraction. Keep replies short (the models are tiny and the
  context window is small).
- Reuse the existing time-seeded xorshift `shuffle`/PRNG already in
  `training.rs` rather than pulling in `rand` (matches the current convention in
  that file — see the `shuffle` comment). Take a count parameter (how many
  examples to emit) and write to `storage::sft_dir()`.
- Keep it plain free functions; no template engine dependency — `format!` is
  enough for this scale (per `backend/CLAUDE.md` "direct over clever").

### Verification
- `cargo build` clean.
- Unit test: generating N examples yields N well-formed JSONL lines that
  `serde_json` round-trips and that `build_training_example` (ISSUE-0031) renders
  to a non-empty masked span each.

### Non-goals
No LLM-in-the-loop generation, no external corpora — purely local templated data.
Richer/self-distilled data is a future issue if wanted.

## Comments
### Auditor · 2026-07-10
The lowest-priority child (P3): the SFT *machinery* (0031–0035, 0037) works with
any conversation JSONL, including hand-written files, so this generator is a
convenience, not a blocker — hence it `blocks` nothing. Kept blocked-by 0031 only
so its round-trip test can assert the generated examples render to a valid mask.
Deliberately templated/verifiable-by-construction so the SFT smoke test has clean
signal rather than noisy synthetic text.
