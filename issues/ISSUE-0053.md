---
id: ISSUE-0053
title: "f32 master weight buffers in GpuWorkspace — one DeviceBuffer per trained weight"
scope: backend/llm
priority: P1
status: todo
rule: none
location: backend/src/llm/src/training/gpu_workspace.rs:11-23 backend/src/llm/src/training/gpu_workspace.rs:109
blocked-by: []
blocks: [ISSUE-0055]
---

## Description
### What
Add an **f32 master buffer** (`DeviceBuffer`, not `DeviceBufferF16`) alongside
each *trained* weight in `GpuWorkspace`, allocated in `GpuWorkspace::new`. This is
pure storage — no kernel or optimizer change here (that is 0054/0055); this child
just makes the masters exist and get allocated/zeroed so later children can wire
them in.

### Which weights get a master
Exactly the params `gpu_adam_update` steps (`transformer.rs:1553`): the two shared
weights `d_w_output` and `d_embedding`, and the ten per-layer weights
`d_wq_all`, `d_wk_all`, `d_wv_all`, `d_wo`, `d_w1`, `d_w2`, `d_b1`, `d_b2`,
`d_ln_gamma`, `d_ln_beta`. **Not** `d_positional` (never updated), and **not** any
activation/scratch buffer. Mirror the existing `Vec<DeviceBufferF16>` /
`DeviceBufferF16` shapes exactly — same element counts (`h*d*hd`, `d*d`, `d*ff`,
`ff*d`, `ff`, `d`, `d*vocab`, `vocab*d`).

### Shape of the change
- Add fields, e.g. `m_wq_all: Vec<DeviceBuffer>`, … `m_w_output: DeviceBuffer`,
  `m_embedding: DeviceBuffer` (pick a prefix that reads clearly against the
  existing `d_` view / `g_` grad / `v_`,`s_` moment prefixes — `m_` collides with
  no existing prefix here; if it reads ambiguously against "moment", use
  `master_` — the Builder decides, but keep it consistent).
- Allocate each in `new` next to its f16 sibling (line ~119–131), same length.
- Include them in the struct literal that `new` returns.
- They can start zeroed (`DeviceBuffer::alloc`); **0056** seeds the real values.

### Out of scope
- No `adam_kernel` change, no `gpu_adam_update` change, no seeding/save change —
  those are 0054/0055/0056. This child must **compile** and leave behaviour
  byte-identical (the masters are allocated but unread), so it lands green on its
  own.

### Verification
- `cargo build` + `cargo clippy` clean from `backend/` (allow a
  `#[allow(dead_code)]` on the new fields if clippy flags them unread until 0055
  wires them — remove it there).
- Existing GPU tests unaffected (`--test-threads=1`): the masters are inert.

## Comments
### Auditor · 2026-07-13
First child of ISSUE-0052 and the only pure-storage step. Kept deliberately inert
so it can't regress anything on its own; the value only flows once 0055 threads
the pointers and 0056 seeds them.
