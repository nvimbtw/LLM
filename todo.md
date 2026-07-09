# todo.md

Single source of truth for outstanding work. Managed by the agent workflow in
[`AGENTS.md`](AGENTS.md): the **Auditor** files issues, the **Fixer** (Claude
Code) resolves them, the **Verifier** confirms. Read the format below before
adding or changing anything.

## Status lifecycle

```
        Auditor files            Fixer starts         Fixer done         Verifier passes
  (nothing) ─────────► todo ──────────────► in-progress ─────► verification ─────────► done
                        ▲                                            │
                        └──────────── Verifier bounces ─────────────┘
                                   (with a reason in verification-notes)
```

- `todo` — filed, not started. **Owner of transition:** Auditor (create), Verifier (bounce back).
- `in-progress` — Fixer is actively working it. **Owner:** Fixer.
- `verification` — fix landed, awaiting check. **Owner:** Fixer (sets it).
- `done` — verified resolved. **Owner:** Verifier.

## Priority legend

- **P0** — critical: broken build, data loss, or a correctness bug that produces
  wrong output in normal use. Drop everything.
- **P1** — high: latent correctness bug (wrong output under some config), or a
  rule violation with real risk. Fix soon.
- **P2** — medium: maintainability/readability/duplication that isn't biting yet.
- **P3** — low: polish, nits, nice-to-haves.

## Scope legend

`frontend` · `backend` · `backend/llm` · `tokenizer` · `data-pipeline` ·
`crawler` · `infra` · `docs` · `cross` (spans several).

## Issue format

Copy this template exactly. One issue per discrete problem. Separate issues with
a `---` rule.

```
## ISSUE-XXXX · <short imperative title>
- **scope:** <from scope legend>
- **priority:** <P0|P1|P2|P3>
- **status:** <todo|in-progress|verification|done>
- **rule:** <Rn from skills/AUDITOR.md, or `none`>
- **location:** `path/to/file.rs:line` (or `:start-end`)
- **blocked-by:** <ISSUE-XXXX, or —>
- **blocks:** <ISSUE-XXXX, or —>

**Description:**
<What's wrong, why it matters, and — if obvious — the shape of the fix.>

**Verification notes:** _(Verifier fills on pass/bounce)_ —
```

---

# Issues

## ISSUE-0001 · Derive inference d_ff from weights, not a hardcoded ×4
- **scope:** backend/llm
- **priority:** P1
- **status:** done
- **rule:** R10
- **location:** `backend/src/llm/src/training/transformer.rs:345`
- **blocked-by:** —
- **blocks:** —

**Description:**
`feed_forward` hardcodes `let d_ff = d_model * 4;`, but the FF width is
`dimensions * ff_multiplier` and `ff_multiplier` is a user-configurable training
field. Training uses the real width; only the inference path assumes 4. The
frontend default is 4, which masks it — but the moment a model is trained with
`ff_multiplier != 4`, inference reads `w1` with the wrong stride and generates
garbage regardless of training quality. Derive `d_ff` from `w1[0].len()` (the
actual weight shape) instead of assuming ×4.

**Verification notes:** Confirmed `let d_ff = w1[0].len();` at line 345 with explanatory comment. `cargo build` clean. —

---

## ISSUE-0002 · Remove dead `cuda_sgd_momentum` FFI (or wire it up)
- **scope:** backend/llm
- **priority:** P3
- **status:** done
- **rule:** R7
- **location:** `backend/src/llm/src/training/transformer.rs:1240`
- **blocked-by:** —
- **blocks:** —

**Description:**
`cuda_sgd_momentum` is declared in the `extern "C"` block but never called (the
optimizer moved to Adam), producing a standing `dead_code` warning. R7 requires a
clean build. Either delete the declaration (and the kernel if unused elsewhere) or
add a justified `#[allow(dead_code)]` with a note if it's kept intentionally.

**Verification notes:** Confirmed `cuda_sgd_momentum` removed from the `extern "C"` block in transformer.rs. `cargo build` shows no related warning. —

---

## ISSUE-0003 · De-duplicate Transformer serialization read/write logic
- **scope:** backend/llm
- **priority:** P2
- **status:** done
- **rule:** R6
- **location:** `backend/src/llm/src/training/transformer.rs:850-1057, 1438-1650`
- **blocked-by:** —
- **blocks:** —

**Description:**
`save_to_path`/`to_bytes` and `load`/`from_bytes` carry near-identical
serialization logic (the same field order, the same 2d/1d read/write helpers,
the same versioned per-layer layout) duplicated across `File` and `Cursor`/`Vec`
targets. They must change together on every format bump — a classic R6 hazard.
Factor the layout into one place parameterized over the reader/writer (e.g. via
`std::io::Read`/`Write` trait objects) so there's a single definition of the file
format.

**Verification notes:** Confirmed `write_model`/`read_model`/`read_header`/`read_qkv_weights`/`read_block_weights`/`read_2d`/`read_1d` shared by file and bytes paths. `save_to_path`/`to_bytes`/`load`/`from_bytes` all delegate. Verified no remaining duplicated serialization code. —

---

## ISSUE-0004 · Split `train_step` — it is ~440 lines (R1)
- **scope:** backend/llm
- **priority:** P1
- **status:** todo
- **rule:** R1
- **location:** `backend/src/llm/src/training/transformer.rs:409-849`
- **blocked-by:** —
- **blocks:** —

**Description:**
`train_step` spans ~440 lines doing embedding forward → per-layer forward pass through all layers → logits + loss → backward pass (layers in reverse) → gradient clipping → embedding gradient accumulation in one body. Each stage block currently prefixed with `// ── N. … ──` should be a named helper (e.g. `embedding_forward`, `forward_pass`, `compute_loss`, `backward_pass`, `clip_gradients`).

**Verification notes:** _(Verifier fills on pass/bounce)_ —

---

## ISSUE-0005 · Split `load` and `from_bytes` — each ~150 lines (R1)
- **scope:** backend/llm
- **priority:** P2
- **status:** done
- **rule:** R1
- **location:** `backend/src/llm/src/training/transformer.rs:909-1057, 1508-1650`
- **blocked-by:** ISSUE-0003
- **blocks:** —

**Description:**
`load` (148 lines) and `from_bytes` (142 lines) contain the same versioned field-reading loop repeated for each weight tensor, well over the R1 limit. After the ISSUE-0003 de-duplication, split the header-parse and per-layer-read stages into named functions.

**Verification notes:** `load` is now 4 lines (delegates to `read_model`), `from_bytes` is 2 lines. Both well under 60. —

---

## ISSUE-0006 · Split `run_training_loop` and `train_epoch` — both exceed 60 lines (R1)
- **scope:** backend/llm
- **priority:** P1
- **status:** todo
- **rule:** R1
- **location:** `backend/src/llm/src/train.rs:194-388`
- **blocked-by:** —
- **blocks:** —

**Description:**
`run_training_loop` is 87 lines and `train_epoch` (extracted from the original 214-line `run_training_loop`) is 107 lines — both still over the R1 60-line limit. Extract the progress callback construction block from `run_training_loop` (lines 344-362) and split `train_epoch` into named phases (e.g. `process_chunk`, `step_optimizer`, `report_progress`).

**Verification notes:** BOUNCED (2nd) — `run_training_loop` unchanged at 87 lines, `train_epoch` at 107 lines also exceeds limit. Neither was addressed since previous bounce. —

---

## ISSUE-0007 · Split `handle_message` — 172 lines (R1)
- **scope:** backend
- **priority:** P2
- **status:** done
- **rule:** R1
- **location:** `backend/src/routes/chat.rs:190-362`
- **blocked-by:** —
- **blocks:** —

**Description:**
`handle_message` spans 172 lines covering parse → resolve chat → persist message → resolve model → build history → spawn inference → stream tokens → persist reply → record usage → send done. Each stage should be a named helper.

**Verification notes:** `handle_message` is now ~35 lines with `parse_frame`, `resolve_chat`, `persist_user_message`, `resolve_model`, `load_history`, `stream_reply`, `finalize_turn` extracted as named helpers. —

---

## ISSUE-0008 · Unwraps on file-parsed model data (R3, R11)
- **scope:** backend/llm
- **priority:** P1
- **status:** todo
- **rule:** R3, R11
- **location:** `backend/src/llm/src/training/transformer.rs:935,944,1533,1543`
- **blocked-by:** —
- **blocks:** —

**Description:**
`read_2d` and `read_1d` closures inside `load` (lines 935, 944) and `from_bytes` (lines 1533, 1543) use `.try_into().unwrap()` on byte slices from a model file. R3 requires a justification why failure is impossible (or use `?`/`expect` instead); R11 explicitly forbids `unwrap` on file data. While `chunks_exact(4)` makes this infallible, add a comment. Same pattern at `tokenize/src/bpe.rs:206-208` (load_merges).

**Verification notes:** BOUNCED — transformer.rs now has justification comments on the unwraps (lines 954, 966), but `tokenize/src/bpe.rs:209-211` still has 3 bare `.try_into().unwrap()` calls in `load_merges` with no comment. The issue explicitly covers that location. —

---

## ISSUE-0009 · Unsafe blocks in `train_step` lack safety comments (R4)
- **scope:** backend/llm
- **priority:** P2
- **status:** todo
- **rule:** R4
- **location:** `backend/src/llm/src/training/transformer.rs:409-849`
- **blocked-by:** —
- **blocks:** —

**Description:**
Every `unsafe { ... }` block in `train_step` (~40 instances across 440 lines) invokes a CUDA FFI function with no comment explaining why the call is sound — which invariants the caller guarantees (buffer sizes, alignment, GPU state). R4 requires each unsafe block to carry a short justification. E.g. `// safe: all d_* buffers are allocated at max_s*d*f16 and s ≤ max_s`.

**Verification notes:** _(Verifier fills on pass/bounce)_ —

---

## ISSUE-0010 · Inline magic hyperparameter values should be named constants (R8)
- **scope:** backend/llm
- **priority:** P1
- **status:** done
- **rule:** R8
- **location:** `backend/src/llm/src/training/transformer.rs:248,459,561,715,797,818,1327-1329`
- **blocked-by:** —
- **blocks:** —

**Description:**
The following tunable values are inlined as literals with no named `const`:
- `1e-5f32` — layer-norm epsilon, 5 places (lines 248, 459, 561, 715, 797)
- `1.0f32` — gradient clip max norm (line 818), duplicated in train.rs:162
- `0.9`, `0.999`, `1e-8` — Adam beta1, beta2, epsilon (lines 1327-1329)
- `3` — early-stopping patience (train.rs:154)
- `4` — gradient accumulation steps (train.rs:158)
- `100` — linear LR warmup steps (train.rs:166)

Each should be a documented `const` at the module top.

**Verification notes:** Confirmed `LAYER_NORM_EPS`, `TRAIN_STEP_CLIP_NORM`, `ADAM_BETA1`, `ADAM_BETA2`, `ADAM_EPS` defined in transformer.rs; `EARLY_STOP_PATIENCE`, `ACCUM_STEPS`, `MAX_GRAD_NORM`, `WARMUP_STEPS` defined in train.rs. All literals replaced. —

---

## ISSUE-0011 · Missing doc comments on public Transformer API surface (R9)
- **scope:** backend/llm
- **priority:** P2
- **status:** done
- **rule:** R9
- **location:** `backend/src/llm/src/training/transformer.rs:49-116, 118-407, 850-1057, 1438-1650`
- **blocked-by:** —
- **blocks:** —

**Description:**
`pub struct Transformer` and its public methods (`init`, `forward`, `forward_seq`, `unembed`, `train_step`, `save_to_path`, `save`, `load`, `to_bytes`, `from_bytes`) lack `///` doc comments. R9 requires every `pub fn`/`pub struct` to document its contract (invariants, panics, units). E.g. `train_step` does not document that it mutates `ws`, or that `chunk` must be ≥2. `load`/`from_bytes` don't mention the binary format versioning or that they panic on `heads=0`.

**Verification notes:** Confirmed `///` doc comments on `Transformer` struct (lines 99-101), `init` (122-123), `forward_seq` (173-175), `train_step`, `save_to_path` (855), `save` (861), `load` (866-867), `to_bytes`, `from_bytes` (1496). —

## ISSUE-0012 · No bounds validation on model file header fields (R10)
- **scope:** backend/llm
- **priority:** P1
- **status:** done
- **rule:** R10
- **location:** `backend/src/llm/src/training/transformer.rs:24-47, 909-1057, 1508-1650`
- **blocked-by:** —
- **blocks:** —

**Description:**
`read_model_header`, `load`, and `from_bytes` read dimensions, heads, layers, etc. from a model file (external input) and immediately use them in arithmetic without validating they are sane positive values. If `heads` is 0, `dimensions / heads` (line 961 / from_bytes equivalent) panics with division by zero. Validate at the boundary: reject any header field ≤ 0 before proceeding.

**Verification notes:** Confirmed `validate_header_fields` (lines 66-97) checks all 6 header fields are positive and that `dimensions % heads == 0`. Called from both `read_header` and `read_model_header`. —

---

## ISSUE-0013 · Forward pass logic duplicated between `forward` and `forward_seq` (R6)
- **scope:** backend/llm
- **priority:** P2
- **status:** done
- **rule:** R6
- **location:** `backend/src/llm/src/training/transformer.rs:118-229`
- **blocked-by:** —
- **blocks:** —

**Description:**
`forward` (lines 118-177) and `forward_seq` (lines 179-229) have a near-identical inner loop: embed → positional → per-layer { ln → attn → residual → ln → ff → residual }. The only difference is that `forward` also produces training targets. Factor the shared forward-pass body into a private helper (e.g. `compute_hidden_states`) parameterized over whether targets are needed.

**Verification notes:** Confirmed `pub async fn forward` has been removed entirely. Only `forward_seq` remains — the duplication is eliminated. —

---

## ISSUE-0014 · `insert_user` error response leaks internal error details (rule: none)
- **scope:** backend
- **priority:** P2
- **status:** done
- **rule:** none
- **location:** `backend/src/routes/user.rs:50-52`
- **blocked-by:** —
- **blocks:** —

**Description:**
When `crypto::password::hash_password` fails, the handler returns the error details to the client via `format!("failed to hash password: {}", e)` instead of a generic server-error message. This violates the CLAUDE.md convention: "Never return raw error internals in the HTTP response body". Log server-side (`eprintln!`) and return a generic `"failed to create user"` message instead.

**Verification notes:** Confirmed line 50 now uses `eprintln!("password hashing failed: {}", e);` for server-side logging, and line 52 returns generic `"failed to create user"` to the client. —
