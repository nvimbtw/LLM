---
id: ISSUE-0034
title: SFT dataset builder — conversations to sft_corpus.bin + sft_mask.bin
scope: backend
priority: P2
status: todo
rule: none
location: backend/src/training.rs:65-177 backend/src/storage.rs:27-62
blocked-by: [ISSUE-0031]
blocks: [ISSUE-0035, ISSUE-0037]
---

## Description
### What
Add `build_sft_dataset(...)` next to `build_dataset` (`backend/src/training.rs`):
read chat conversations, render each with ISSUE-0031's `build_training_example`,
and write two aligned files the SFT loop consumes:
- `datasets/sft_corpus.bin` — flat little-endian `u32` token ids (same format as
  `corpus.bin`, so `chunks_exact(4)` reads it).
- `datasets/sft_mask.bin` — one byte per token, `1`=train / `0`=ignore, aligned
  1:1 with `sft_corpus.bin`.

### Why
The SFT loop (ISSUE-0035) needs chat-structured tokens **and** a mask; the current
`build_dataset` produces neither — it concatenates raw prose and trains a fresh
BPE tokenizer. SFT is a *fine-tune*, so it must reuse the base model's existing
tokenizer, not build a new one.

### The critical correctness constraint (call this out loudly)
**SFT must reuse the base model's `merges.bin`, never retrain BPE.** The token
ids the base model learned are fixed; re-training the tokenizer would renumber the
vocab and the special-marker ids, and the fine-tune would be gibberish. So
`build_sft_dataset` **loads** the base model's merges (via
`tokenize::bpe::load_merges` from the base model's dir) and passes them to
`build_training_example` — it does not call `tokenize::bpe::train`.

### Shape of the fix
- **Input format:** conversations as JSONL, one object per line, e.g.
  `{"system": "...", "turns": [{"role":"user","text":"..."},
  {"role":"assistant","text":"..."}, ...]}`. Add a `sft_dir()` to `storage.rs`
  (`raw/sft`) for these inputs; the synthetic generator (ISSUE-0036) also writes
  here. Parse with `serde_json` (already a dependency).
- For each conversation: call `build_training_example(system, turns, &merges)` →
  `(ids, mask)`. Append to the flat corpus/mask. **Between examples**, append an
  `<|endoftext|>` (U+E000) token with mask `0`, so the model sees a boundary but
  is never trained to predict across two unrelated conversations.
- Write via a `save_corpus`-style helper for the ids and a plain
  `fs::write(&mask_bytes)` for the mask. Return an SFT-flavoured stats struct
  (examples, total tokens, trainable tokens, % trainable) for the dashboard +
  `meta.json` — trainable% is a useful sanity signal (should be well under 50%).
- Keep it plain free functions in the same file/style as `build_dataset`; no new
  trait or abstraction (per `backend/CLAUDE.md`).

### Verification
- `cargo build` (workspace) clean.
- Unit test: a 2-conversation fixture produces a corpus whose length equals the
  sum of the rendered example lengths + boundary tokens, a mask of equal length,
  and a trainable-token count matching the assistant spans.

## Comments
### Auditor · 2026-07-10
The storage/IO half of the SFT data path — deliberately in the `backend` crate
(where `storage.rs` owns the layout), distinct from the pure format logic in
ISSUE-0031 which lives in the `llm` crate. The "reuse the base tokenizer, do NOT
retrain BPE" constraint is the single most important correctness note in the whole
epic; a Builder who misses it will produce a dataset that silently trains on the
wrong vocab. Blocked-by 0031 (it calls `build_training_example`); feeds both the
loop (0035) and the orchestration (0037).

### VCS · 2026-07-10
Committed as `d5b9a57` on branch `dev`.
