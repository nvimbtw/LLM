---
id: ISSUE-0093
type: issue
title: LN2 backward writes out of bounds when ctx_window exceeds 2 * dimensions
scope: backend/llm
priority: P0
status: done
rule: R10
location: backend/src/llm/src/training/gpu_workspace.rs:314 backend/src/llm/src/training/transformer/backward.rs:116
blocked-by: []
blocks: []
---

## Description

`b_dd` was allocated as `Buffer::alloc(d * d)` — `d*d` **f32**, i.e. `4*d*d`
bytes — and used for two things with different element types:

- `backward.rs:161` — the `wo` gradient, f32, `d*d` elements. Fits exactly.
- `backward.rs:116` — the `dx` output of `cuda_layer_norm_backward` (LN2's
  backward). That parameter is `__half *dx` (`shaders.cu:618`) and the kernel
  writes `rows * cols` = `s*d` **halves**, i.e. `2*s*d` bytes
  (`row_dx = dx + row * cols`, then `row_dx[j]` for `j < cols`).

The f16 use fits only while `2*s*d <= 4*d*d`, i.e. **`s <= 2*d`**. Nothing
checked it, and the sizing expression (`d * d`) did not mention `max_s` at all,
so the constraint was invisible at the allocation site.

At `d = 256`:

- `ctx_window` 512 → `s = 512 = 2*d`. Exactly at the ceiling, no overrun.
  Shakedown 1 and every run before it were at this shape.
- `ctx_window` 1024 → `s = 1024`. Writes **524,288 bytes into a 262,144-byte
  allocation**, once per layer per step, for the whole run. Balanced 1
  (`backend/data/artifacts/models/af6d9ac8-d3f7-4994-8021-efdcd72d484e`) trained
  entirely in this state.

**This corrupts training, it is not merely undefined behaviour.** Measured with
a controlled A/B in a single process — same binary, same allocation layout, same
deterministic weights and input, the only difference being whether LN2's `dx`
goes out of bounds (d 256, ctx 1024, L4, lr 1e-4, 8 steps):

| step | in bounds | out of bounds |
|---|---|---|
| 1 | 9.044299 | 9.044297 |
| 2 | 8.904865 | 8.904864 |
| 3 | 8.777497 | 8.780521 |
| 6 | 8.433416 | 8.547108 |
| 7 | 8.330494 | **8.639627** |
| 8 | 8.232923 | 8.556607 |

Run-to-run noise is ~1e-5 (the LN backward accumulates gamma/beta with
`atomicAdd`), so the step-8 gap of 0.32 nats is ~30,000x the noise floor. The
out-of-bounds run also stops descending monotonically — it rises at step 7,
which is the signature that made this look like a bad hyperparameter rather than
a bug. At `ctx 512` the two are identical to 1e-5, which is the control.

The overrun does **not** land in `b_dff[0]`, despite that being the next
allocation in `Workspace::new`. It lands in a non-workspace allocation — the
file-static `attn` / `ff` scratch in `shaders.cu`, which is `cudaMalloc`'d
lazily on the first step, after the workspace. `attn` is the one that matters:
it is live *within the same layer's backward*, since `attn_sublayer_backward`
runs immediately after `ffn_sublayer_backward` and
`cuda_attention_backward_batched` reads and writes `attn.dQ/dK/dV/dP/dS`. So the
LN2 overrun corrupts that layer's attention gradients on the very next call.

Which allocation gets hit is allocator-dependent and varies between processes
and shapes — probing a workspace before the first step shows the span empty.
That is exactly why it is dangerous: "it seemed to train" is not evidence of
safety.

## Comments

### Builder · 2026-07-29

Split the two uses. `b_dd` was one buffer serving two unrelated purposes with
two different element types, which is the whole root cause — the name said
nothing about either, so neither constraint was visible where it was allocated.

- `b_dd` → **`b_dwo`**, `Buffer::alloc(d * d)` f32, sole purpose the `wo`
  gradient scratch. Named for what it holds now.
- New **`ln2_dgrad`**, `DeviceBufferF16::alloc(max_s * d)`, for LN2's `dx`. This
  makes it symmetric with LN1's backward, which has always written into
  `b_sd[1]` — an f16 buffer sized at `max_s`. LN1 was correct all along; only
  LN2 borrowed the wrong buffer.

Cost is one `max_s * d` f16 allocation: 512 KB at d 256 / ctx 1024.

Added `Workspace::assert_scratch_fits`, run once in `new`, checking in **bytes**
that every scratch a kernel fills with an `[s, d]` block holds one at `max_s`.
It is a hard panic, not a `debug_assert`: the failure it catches is an
out-of-bounds device write, which does not fault, does not stop the loss
descending, and silently corrupts whatever the driver placed next. Verified
non-vacuous by shrinking `ln2_dgrad` to `d * d` and confirming it fires with the
buffer name and both sizes.

Audited every other scratch buffer for the same mismatch while here. All clean:
`b_sd`/`b_sd_head`/`d_wo_x`/`d_dhidden` are f16 at `max_s * d`; `b_dff`/`b_ff`
are f32 sized by `d`/`ff` and only ever written as f32; `b_d` is f32 `[d]`;
`d_head_ptrs` is `Buffer::alloc(h * 2)` = `8h` bytes holding exactly `h`
eight-byte pointers (odd, since it is an f32 buffer holding pointers, but
correctly sized — noted for the Auditor, not changed).

Tests: `tests/scratch_bounds.rs`, four of them, at `ctx = 4 * d`. They check the
sizing arithmetic rather than trying to observe the corruption, because the
corruption is invisible from the loss. One of them pins the two real pretrain
shapes against the old ceiling so the boundary is documented in something
executable.

`compute-sanitizer` is present at `/opt/cuda/bin` but cannot attach under WSL2
("Failed to initialize WDDM debugger interface"), so it was not usable here.
The A/B above is what stands in for it.

Ran: `cargo build` (clean), `cargo clippy -p llm --all-targets` (zero warnings),
`cargo test -p llm -- --test-threads=1` (183 + 4 + 3 + 5 pass), and the ignored
GPU guards — `attention_learns_to_copy` and all three `bench_step` shapes pass.

**Consequence for the models on disk:** Balanced 1 is invalid and must be
re-pretrained; this is a second, independent reason on top of TODO-0092. Every
model at `ctx <= 2 * d` — Shakedown 1 at ctx 512 / d 256, and the whole
Shakedown 1.01 SFT/KTO chain built on it — is unaffected, since `s = 2*d` sits
exactly at the ceiling with no overrun.

### VCS · 2026-07-29 · 2f0bb0a on dev

### Verifier · 2026-07-29 · pass

**Verified.** Checked the following:

- **Code fix at location:** `backend/src/llm/src/training/gpu_workspace.rs:314-315` — `b_dd` split into `b_dwo` (f32 `d*d` for wo gradient) and `ln2_dgrad` (f16 `max_s*d` for LN2's dx). `backward.rs:116-129` uses `ws.ln2_dgrad` in place of old `b_dd`. No remaining references to `b_dd` in source.
- **`assert_scratch_fits`** at `gpu_workspace.rs:443-471` checks every f16 `[s,d]` scratch in bytes against `max_s * d * sizeof(f16)`, plus `b_dwo` against `d*d`. Hard panic, not `debug_assert`.
- **Tests:** `tests/scratch_bounds.rs` — 4/4 pass, including an end-to-end `train_step` at `ctx = 4*d` (well past the old `s <= 2*d` ceiling) and a pin of the two real pretrain shapes.
- **Full test suite:** 183 unit tests + 3 kernel_sanity + 5 train_test + 4 scratch_bounds = 195 pass. GPU-guarded tests (attention_learns, bench_step) unchanged.
- **Build:** `cargo build -p llm` clean. `cargo clippy -p llm --all-targets` — zero warnings.
- **No collateral violations** found — naming, sizing, and element types all consistent.
