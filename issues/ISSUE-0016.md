---
id: ISSUE-0016
title: Unsafe blocks in device_buffer.rs and gpu_workspace.rs lack safety comments (R4)
scope: backend/llm
priority: P1
status: done
rule: R4
location: backend/src/llm/src/training/device_buffer.rs:19-21,30-32,38-44,50-56,64-66,79-81,92-98,104-110,115-121,128-130 backend/src/llm/src/training/gpu_workspace.rs:211,257-276,294-296
blocked-by: []
blocks: []
---

## Description
Every `unsafe` block in `device_buffer.rs` (`DeviceBuffer::alloc`, `alloc_zeroed`,
`from_slice`, `to_vec`, `drop`, and `DeviceBufferF16` equivalents) and in
`gpu_workspace.rs` (`cublas_init`, `zero_accumulators` inline-CUDA-redecl,
`cublas_shutdown`) lacks a safety comment justifying why the raw pointer operations
are sound. ISSUE-0009 covers the `train_step` unsafe blocks in `transformer.rs`;
this covers the remaining unsafe blocks in the GPU-layer files.

## Comments
### Builder · 2026-07-10
Added a `// SAFETY: …` comment directly above every `unsafe` block cited in the
description, in both files. R4 asks for the invariants the caller/context
guarantees that make the raw pointer op sound — so each comment states, in terms
specific to that call, why the pointer is valid and why the byte count passed to
the FFI call matches the allocation:

- `device_buffer.rs` (`DeviceBuffer` + `DeviceBufferF16`): `alloc` — output
  pointer is a valid local out-param sized by the same `len` used everywhere else;
  `alloc_zeroed`/`from_slice`/`from_f32_slice`/`upload_f32_slice` — the device ptr
  was just allocated (or is guaranteed by construction) for exactly the byte count
  passed to `cudaMemset`/`cuda_memcpy_h2d`, and the host-side slice/`Vec` is a live
  allocation of the matching length; `to_vec`/`to_f32_vec` — the destination `Vec`
  is freshly allocated with `self.len` elements, matching the device buffer's
  invariant size; `Drop` — free happens exactly once since `drop` runs at most
  once per value.
- `gpu_workspace.rs`: `cublas_init()` in `GpuWorkspace::new` — no-arg FFI call,
  sound because this training loop is single-threaded so cuBLAS handle init isn't
  raced, and it's paired 1:1 with `cublas_shutdown()` in `Drop`; `zero_accumulators`
  — every `cudaMemset` count matches the element count that buffer was allocated
  with in `GpuWorkspace::new` (same `self.d`/`self.vocab`/`self.h`/`self.hd`/
  `self.ff`/`self.max_s` dims), scaled by 4 bytes/f32; `cublas_shutdown()` in
  `Drop` — mirrors the init pairing.

No logic changed, only comments added — this is purely a documentation fix per R4
(unlike ISSUE-0009, which covered the analogous `train_step` blocks in
`transformer.rs` and is already `done`).

Ran: `cargo build -p llm` — finished clean, zero warnings.

### Verifier · 2026-07-10 · pass
Confirmed every unsafe block cited in the description now has a `// SAFETY: …`
comment in both files:

**`device_buffer.rs`** (10 blocks): `DeviceBuffer::alloc`, `alloc_zeroed`,
`from_slice`, `to_vec`, `Drop`; `DeviceBufferF16::alloc`, `from_f32_slice`,
`upload_f32_slice`, `to_f32_vec`, `Drop` — each comment states why the pointer
is valid and why the byte count matches the allocation, as required by R4.

**`gpu_workspace.rs`** (3 blocks): `cublas_init` (line 216), `zero_accumulators`
(lines 268–287), `cublas_shutdown` (line 309) — each comment explains the
soundness invariants (single-threaded guarantee for cuBLAS handle init/shutdown,
buffer dims matching allocation dims for `cudaMemset`).

No logic changed — comments only. `cargo build -p llm` clean, zero warnings. No
new `.unwrap()`/`.expect()` introduced. One minor observation: `cudaMemset`
returns an `i32` status code that is silently ignored in both files (pre-existing,
not part of this issue). The fix resolves the described problem cleanly.

### VCS · 2026-07-10
Committed at `b809b62` on branch `dev`
