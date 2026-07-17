---
id: ISSUE-0034
type: issue
title: SFT dataset builder — conversations to sft_corpus.bin + sft_mask.bin
scope: backend
priority: P2
status: done
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
### Builder · 2026-07-10
Added `build_sft_dataset(base_model_dir)` + supporting types to
`backend/src/training.rs`, and `storage::sft_dir()` (`raw/sft`).

**The critical constraint is honoured: the base tokenizer is loaded, never
retrained.** `build_sft_dataset` reads `base_model_dir/merges.bin` via
`tokenize::bpe::load_merges` and passes those merges to
`llm::chat_template::build_training_example`. It never calls `tokenize::bpe::train`,
so the token ids (and the special-marker ids) stay exactly what the base model
learned.

**What it does:** reads every `*.jsonl` in `sft_dir()` (sorted for determinism),
parses one `SftConversation` per line (`{system?, turns:[{role,text}...]}` via
serde), renders each with `build_training_example`, and concatenates into a flat
`corpus: Vec<u32>` + `mask: Vec<u8>`. Between conversations it inserts one
`<|endoftext|>` token (encoded via the base merges) with mask `0`, so the model
sees a boundary but is never trained across two unrelated conversations. Writes
`datasets/sft_corpus.bin` (via `save_corpus`, the same u32-LE format
`llm::sft_train` will read with `chunks_exact(4)`) and `datasets/sft_mask.bin`
(raw bytes, 1:1 with the corpus). Returns `SftDatasetStats { conversations,
total_tokens, trainable_tokens }` for the dashboard/meta.

**Decisions worth noting:**
- Unknown role strings are a **hard error** (`parse_role`), not silently skipped —
  bad data should fail loudly rather than train on a wrong mask.
- Empty SFT dir / empty output are errors (`NotFound` / `InvalidData`), matching
  `build_dataset`'s "no files" behaviour.
- The new items carry `#[allow(dead_code)]` with a comment: they're exercised by
  tests now and wired into the run/route by ISSUE-0037; this mirrors the
  "defined ahead of its caller" convention already used with `#![allow(dead_code)]`
  in `storage.rs`. Keeps R7 (clean build) satisfied without leaving them unused-and-
  warning.

**Verification.**
- `cargo test -p backend sft_tests -- --test-threads=1` → **2 passed**:
  - `test_build_sft_dataset_aligns_and_counts`: two conversations (incl. one with
    a system prompt) → `sft_corpus.bin` length is exactly `4 × sft_mask.bin`
    length (u32 vs u8 per token), `total_tokens` matches the mask length, and
    `trainable_tokens` equals the sum of the two examples' assistant-span mask
    counts (computed independently via `build_training_example`).
  - `test_build_sft_dataset_rejects_unknown_role`: `parse_role` rejects `system`,
    accepts `user`/`assistant`.
- `cargo build -p backend` → clean, **0 warnings**.
- No GPU involved (pure IO + the CPU `build_training_example`).

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

### Verifier · 2026-07-10 · pass

**What was checked:**
- Read `backend/src/training.rs` (lines 246–468) and `backend/src/storage.rs` (lines 39–41).
- `build_sft_dataset` loads the base model's tokenizer via `tokenize::bpe::load_merges(base_model_dir/merges.bin)` — never calls `tokenize::bpe::train`. The critical constraint is honoured.
- Between conversations, inserts `<|endoftext|>` with mask `0` (line 374–377).
- Writes `sft_corpus.bin` (u32-LE via `tokenize::bpe::save_corpus`) and `sft_mask.bin` (raw bytes, 1:1 alignment) to `datasets/`.
- Returns `SftDatasetStats { conversations, total_tokens, trainable_tokens }`.
- Unknown roles are a hard error (`parse_role`, line 291–300).
- Empty SFT dir / empty output are properly rejected with `io::Error`.
- `#[allow(dead_code)]` with justified comments on SFT items (same pattern as `storage.rs`).

**Build & tests:**
- `cargo build -p backend` → clean, **0 warnings**.
- `cargo test -p backend sft_tests -- --test-threads=1` → **2 passed**:
  - `test_build_sft_dataset_aligns_and_counts`
  - `test_build_sft_dataset_rejects_unknown_role`

**Minor note (not blocking):** `cargo clippy -p backend` flags line 376 with `clippy::manual_repeat_n` — `std::iter::repeat_n(0u8, eot.len())` is preferred over `std::iter::repeat(0u8).take(eot.len())`. Not a correctness issue; can be addressed in a follow-up if desired.

### Builder · 2026-07-10
Addressed the Verifier's clippy note: switched the boundary-mask fill to
`std::iter::repeat_n(0u8, eot.len())`. `cargo clippy -p backend` now clean; sft
tests still pass.

### VCS · 2026-07-10
Committed as `715883c` on branch `dev`.
