---
id: ISSUE-0031
title: Render a full conversation to (token_ids, loss_mask) in chat_template
scope: backend/llm
priority: P2
status: done
rule: none
location: backend/src/llm/src/chat_template.rs:63-140
blocked-by: []
blocks: [ISSUE-0034, ISSUE-0035, ISSUE-0036]
---

## Description
### What
Add a sibling to `build_prompt` that renders a **complete** conversation — one
that *includes* the assistant replies — into a token stream **plus a parallel
loss mask**. Proposed shape:

```rust
/// Render a full training example: the whole conversation including assistant
/// replies, returning the token ids and a parallel mask where mask[i] == 1 iff
/// the token predicted from position i (i.e. token i+1) is part of an assistant
/// reply (its content through the closing <|/assistant|> marker, inclusive).
pub fn build_training_example(
    system: Option<&str>,
    turns: &[Turn],            // alternating User/Assistant, ending on Assistant
    merges: &[((u32, u32), u32)],
) -> (Vec<u32>, Vec<u8>);      // (token_ids, mask) — same length
```

### Why
`build_prompt` renders for *inference*: it ends on an **open** `<|assistant|>` for
the model to continue, and it drops historical assistant reasoning. SFT needs the
opposite — the assistant turns are the *training target*, so they must be present
in the token stream, and each position must be tagged trainable/not so the loss
(ISSUE-0032/0033) can ignore everything that isn't the assistant speaking. Putting
this in `chat_template.rs` keeps the module's stated invariant intact: the format
the model is *trained* on and *prompted* with comes from one file and cannot drift.

### Shape of the fix / details
- Reuse `format_turn` so the rendered bytes are byte-identical to what inference
  builds. Do **not** strip thinking from the assistant turns here — during SFT we
  *do* want the model to learn to produce `<|thinking|>…<|/thinking|>` (unlike the
  history compaction `build_prompt` does). Note this divergence in a comment.
- The mask is over **prediction positions**: `mask[i] = 1` iff token `i+1` is
  inside an assistant reply span. Concretely, walk the assembled token stream and
  find the `<|assistant|>` (U+E006) → `<|/assistant|>` (U+E007) spans *after BPE
  encoding*. For every target token from "first token after `<|assistant|>`"
  through "`<|/assistant|>`" inclusive, set the mask at the *preceding* position.
  The `<|assistant|>` opening marker itself is context (mask 0); the closing
  marker is a target (mask 1) so the model learns to stop.
- Encode the whole conversation in **one** `encode(...)` pass (same reasoning as
  `build_prompt`'s single-pass comment — piecemeal encoding risks cross-boundary
  BPE merges). Then locate the marker token ids in the encoded stream to place the
  mask. The marker chars survive as distinct single tokens; confirm how they land
  as ids and derive the mask from those id positions rather than from the string.
- Add unit tests mirroring the existing `chat_template` tests: a one-turn
  conversation should have mask==0 across the system/user/scaffold prefix and
  mask==1 exactly across the assistant reply + closing marker; a two-turn
  conversation should have two masked spans and masked-off user text in between.

### Non-goals
No GPU code, no dataset I/O — this is a pure function over strings→(ids, mask),
unit-testable on CPU. The dataset builder (0034) and synthetic generator (0036)
consume it; the loss (0032/0033) consumes the mask shape it defines.

## Comments
### Builder · 2026-07-10
Added `build_training_example(system, turns, merges) -> (Vec<u32>, Vec<u8>)` to
`backend/src/llm/src/chat_template.rs`, plus three unit tests.

**Key discovery that shaped the implementation.** The issue suggested finding the
`<|assistant|>`/`<|/assistant|>` marker token ids in a single whole-string
encoding and deriving the mask from their positions. That turns out not to be
robust: the pre-tokenizer (`tokenize::pretokenize`) only splits words on **spaces**
(`split_words` splits on `" "`; `split_punctuation` never breaks on the marker's
PUA char). So a marker **glues** to the text adjacent to it inside a single word —
`<|assistant|>Hello` is one pre-tokenizer word, and BPE can merge the marker's
bytes together with the reply's first bytes into one token. There is therefore no
clean token boundary at a marker in a single-pass encoding, and "the marker's
token id" isn't a reliable landmark.

**What I did instead.** Build the token stream *compositionally* from role-tagged
segments and encode each segment on its own:
- system block → not trained
- each user turn → not trained
- each assistant turn → split into three segments: the opening `<|assistant|>`
  cue (not trained), the reply body (**trained**), and the closing
  `<|/assistant|>` (**trained**, so the model learns to stop).

Then I record, per emitted token, whether it came from a target segment, and
shift that onto prediction positions: `mask[i] = target[i+1]` (the final position
has no successor, so it's never trained). This is exactly the epic's convention
(ISSUE-0030).

**Why compositional encoding is the right call, not a compromise.** Splitting at
the marker boundaries makes the reply tokens *stand alone* — which is precisely
how the model sees them at generation time: `infer.rs` generates one standalone
token at a time after the prompt's open `<|assistant|>`. So this matches the
autoregressive path *better* than a glued whole-string encoding would, and it
makes the mask exact. The documented cost is that the concatenated tokenization
can differ by ~1 token at each seam vs. one `encode()` of the full string. I wrote
this reasoning into the function's doc comment so it doesn't get "optimized" back
into a single-pass encode later. (Separately: the marker-gluing is a latent
tokenizer wart — markers aren't atomic tokens — that also causes a minor
train/inference seam mismatch at the reply's *end*. That's pre-existing and out of
scope here; worth an Auditor issue against the tokenizer if we want markers to be
guaranteed-atomic single tokens.)

Did not strip `<|thinking|>` from assistant turns (unlike `format_turn`), per the
issue — during SFT we want the model to learn to produce reasoning. Noted in the
doc comment.

**Verification.**
- `cargo test -p llm chat_template` → 6 passed (3 new: exact trained-token count
  equals `encode(reply)+encode(<|/assistant|>)` lengths; two-turn conversations
  produce exactly two masked spans with the intervening user turn masked off;
  the system/user prefix is entirely `0`).
- `cargo build -p llm` → clean, **0 warnings**.
- The GPU-backed `transformer` tests (`test_embedding_forward_cpu_vs_gpu`,
  `test_train_step_loss_reasonable`, `test_overfits_single_sequence`) crash
  (SIGSEGV/SIGABRT) in this environment — but they do so **identically on the
  committed code with my change stashed**, i.e. this WSL2 box has no functioning
  CUDA device. My change is pure CPU string logic and touches none of that path.
  A GPU-equipped Verifier should see them pass (as they did for ISSUE-0004/0016).

### Auditor · 2026-07-10
Foundational piece of the SFT epic (ISSUE-0030): it pins down the exact mask
convention every downstream issue relies on, and it's fully CPU-testable, so it's
the safest place to start. Kept separate from the dataset builder (0034) because
this is pure format logic that belongs beside `build_prompt`, whereas 0034 is
storage/IO plumbing in the `backend` crate.

### Verifier · 2026-07-10 · pass
**Verdict: pass.** The fix is complete and clean.

**What was checked:**
- `backend/src/llm/src/chat_template.rs` — `build_training_example` present at lines 145-196 with the required signature `(Option<&str>, &[Turn], &[((u32, u32), u32)]) -> (Vec<u32>, Vec<u8>)`.
- **Mask correctness:** `mask[i] == 1` iff token `i+1` is inside an assistant reply span (`<|assistant|>` opening is context/mask 0; reply body + closing `<|/assistant|>` are mask 1; final position with no successor is 0).
- **Thinking preserved:** Assistant turns keep `<|thinking|>` blocks verbatim (divergence from `format_turn` noted in doc comment per issue requirements).
- **Compositional encoding:** Segments are encoded independently with a well-documented rationale (marker gluing in pre-tokenizer makes marker-token-id landmarks unreliable in single-pass encoding).
- **Three unit tests** cover: exact trained-token count for one-turn; two masked spans with intervening user turn masked off; system+user prefix entirely masked off.

**Build & tests:**
- `cargo build -p llm` — clean, 0 warnings ✅
- `cargo clippy -p llm` — clean ✅
- `cargo test -p llm chat_template` — 6/6 passed ✅
- `cargo build` (workspace) — clean ✅

**Rules check (skills/RULES.md):** No violations found. Function body ~51 lines (R1 ✓), all loops bounded (R2 ✓), no dropped `Result`/`unsafe`/magic numbers (R3/R4/R8 ✓), public fn has doc comment (R9 ✓), zero clippy warnings (R7 ✓).

**GPU test note:** The GPU-backed tests (SIGSEGV/SIGABRT) fail identically with the change stashed — pre-existing CUDA-less environment issue, not a regression from this change.
