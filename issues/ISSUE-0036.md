---
id: ISSUE-0036
title: Synthetic chat-data generator for SFT
scope: backend
priority: P3
status: done
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

### VCS · 2026-07-10
Committed as `d5b9a57` on branch `dev`.

### Builder · 2026-07-11
Added the templated synthetic-data generator in `training.rs`.

**What changed:**
- **Shared PRNG (R6):** the xorshift64 that was inlined inside `shuffle` is now two
  named helpers, `rng_seed()` (time-seeded, `| 1` so it's never zero) and
  `xorshift_next(&mut u64)`. `shuffle` calls them, and so does the generator —
  one PRNG, no copy. Still no `rand` dependency, matching the file's convention.
- **`mod synthetic`:** `generate_sft_examples(count)` writes `count` templated
  conversations as one `synthetic.jsonl` into `storage::sft_dir()` (the dir
  `build_sft_dataset` reads). Four verifiable-by-construction categories chosen by
  the PRNG — `arithmetic_example` (+/−/×, subtraction ordered so the answer is
  never negative), `sort_example` (3–5 small numbers → ascending), `capital_example`
  (a small `CAPITALS` table), `echo_example` (repeat a word N times) — each a small
  free function returning `(question, answer)`. `make_example` wraps one in a
  `{system?, turns:[user, assistant]}` conversation, prepending a system prompt on
  ~1/`SYSTEM_PROMPT_ODDS` of them. `GenConversation`/`GenTurn` are the serialize
  twins of the deserialize-only `SftConversation`/`SftTurn`, with
  `skip_serializing_if` so `system` is omitted when absent. Plain `format!`, no
  template engine (`backend/CLAUDE.md` "direct over clever").
- **R7/dead-code:** nothing wires the generator to a runtime trigger yet (a
  dev/admin route or CLI is a future issue, and the issue scopes this to
  `training.rs`/`storage.rs` only), so in this *bin* crate its items are unused. I
  wrapped the section in `#[allow(dead_code)] mod synthetic` — one justified,
  commented allow rather than ten scattered ones — exactly the state
  `build_sft_dataset` sat in before ISSUE-0037 wired it. Serialized-input struct
  can only fail serialization on an allocator fault, but I propagate the `?`
  anyway rather than `unwrap` (R3).

**Verified:**
- `cargo build` / `cargo clippy` — clean, zero warnings (clippy nudged the
  system-prompt modulo to `.is_multiple_of(...)`, applied).
- New test `test_generate_sft_examples_are_well_formed`: generates 50 examples,
  asserts 50 JSONL lines, each round-trips through `SftConversation`, ends on an
  assistant turn, and renders via ISSUE-0031's `build_training_example` to a
  non-empty token sequence **with a non-empty masked (assistant) span** — the
  round-trip-to-valid-mask check the Auditor asked for. 5/5 `sft_tests` pass.
- Added a `static ENV_LOCK: Mutex<()>` in the test module: `DATA_DIR` is a
  process-global that `storage::root()` reads, and there are now two env-touching
  tests. cargo runs tests multi-threaded, so I serialize those two through the
  lock (and renamed their temp dirs so they no longer collide on `process::id()`).
  The pre-existing test's `// single-threaded test` SAFETY note is now actually
  guaranteed rather than assumed.

### Verifier · 2026-07-11 · pass

**What was verified:**

1. **`cargo build`** — clean, zero warnings. ✓
2. **`cargo clippy`** — clean, zero warnings. ✓
3. **`cargo test -p backend training::sft_tests`** — all 5 tests pass, including the new `test_generate_sft_examples_are_well_formed` (generates 50, asserts 50 valid JSONL lines, each round-trips through `SftConversation`, ends on assistant, renders to non-empty masked span via `build_training_example`). ✓
4. **Code review of `mod synthetic` in `training.rs`:**

   | Check | Status |
   |---|---|
   | PRNG extraction (`rng_seed` + `xorshift_next`) as named helpers, `shuffle` refactored to use them (R6) | ✓ |
   | `generate_sft_examples(count)` writes `count` templated conversations to `storage::sft_dir()` as `synthetic.jsonl` | ✓ |
   | Four categories (arithmetic, sort, capital, echo) with programmatically-known correct answers | ✓ |
   | `#[allow(dead_code)] mod synthetic` with justifying comment (R7) | ✓ |
   | No `unwrap()` / `let _ =` / dropped `Result` in non-test code (R3) | ✓ |
   | `serde_json::to_string` error propagated via `?` | ✓ |
   | Named constants: `SYSTEM_PROMPT_ODDS`, `CAPITALS`, `WORDS` (R8) | ✓ |
   | `ENV_LOCK: Mutex<()>` serializes the two env-touching tests | ✓ |
   | No unsafe in non-test code; test `unsafe` has SAFETY comments (R4) | ✓ |
   | All functions well under 60-line limit (R1); all loops bounded (R2) | ✓ |

5. **No regressions:** the `shuffle` refactor preserves Fisher-Yates with the same xorshift64 algorithm. The existing `build_dataset` / `build_sft_dataset` / `run_sft_training` functions are unchanged.

**Verdict:** Pass — all criteria from the issue description and all applicable coding rules are satisfied.
