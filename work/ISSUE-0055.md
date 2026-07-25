---
id: ISSUE-0055
type: issue
title: "Thread f32 master pointers through gpu_adam_update"
scope: backend/llm
priority: P1
status: todo
rule: none
location: backend/src/llm/src/training/transformer.rs:1553-1589
blocked-by: [ISSUE-0053, ISSUE-0054]
blocks: [ISSUE-0056]
---

## Description
### What
Wire the f32 master buffers (0053) into the new `cuda_adam` signature (0054) at
every call in `gpu_adam_update` (`transformer.rs:1553`). The `adam!` macro gains
the master pointer:

```rust
macro_rules! adam {
    ($g:expr, $wm:expr, $wv:expr, $m:expr, $v:expr, $n:expr) => {
        cuda_adam($g, $wm, $wv as *mut c_void, $m, $v, gs, lr, beta1, beta2, eps, bc1, bc2, $n as i32)
    };
}
```

and each of the 12 invocations passes its master alongside its f16 view, e.g.:

```rust
adam!(ws.g_dw_output.ptr, ws.m_w_output.ptr, ws.d_w_output.ptr,
      ws.v_dw_output.ptr, ws.s_dw_output.ptr, d * vocab);
```

### Shape of the change
- One-for-one: for every existing `adam!(...)` line, add the matching
  `ws.m_*.ptr` master argument. The order/length list (2 shared + 10×layers) is
  already correct — do not add/remove weights.
- Remove any `#[allow(dead_code)]` left on the master fields by 0053 (they are now
  read).
- No math change — this is purely plumbing.

### Verification
- `cargo build` + `cargo clippy` clean.
- The GPU trainer tests still pass (`--test-threads=1`). **They will only be
  fully correct once 0056 seeds the masters** — until then the masters are zero,
  so a run started here would drive weights toward `-lr·update` from 0. Therefore
  0055 lands **build-green** but the behavioural assertion belongs to 0056/0057;
  note this explicitly for the Verifier so it doesn't expect a working train run
  from this child alone. (If the Builder prefers, 0055 and 0056 land in the same
  Verifier round — the seed is what makes the wired path meaningful.)

## Comments
### Auditor · 2026-07-13
Pure plumbing between the storage child and the kernel child. Called out that a
working train run isn't provable until masters are seeded (0056), so the Verifier
checks compile + test-still-green here and the numerical win at 0057.
