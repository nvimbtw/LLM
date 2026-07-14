---
id: ISSUE-0062
title: "Rename training::device_buffer::DeviceBuffer to training::device_buffer::Buffer"
scope: backend/llm
priority: P2
status: todo
rule: R21
location: backend/src/llm/src/training/device_buffer.rs:11
blocked-by: []
blocks: [ISSUE-0060]
---

## Description

`training::device_buffer::DeviceBuffer` stutters — the module name `device_buffer` and the type name `DeviceBuffer` repeat each other. Per **R21**, either collapse the module (if it only holds this one public type) or rename the struct to `Buffer` so the path reads `training::device_buffer::Buffer`.

Since `device_buffer.rs` also defines `DeviceBufferF16` (which does NOT stutter — the `F16` suffix adds meaning), renaming `DeviceBuffer` → `Buffer` is the cleanest approach. The module is not single-type, so collapsing it is inappropriate.

### Scope of change
- `backend/src/llm/src/training/device_buffer.rs`: rename `pub struct DeviceBuffer` → `pub struct Buffer`, update all `impl DeviceBuffer` blocks
- `backend/src/llm/src/training/gpu_workspace.rs`: update references to `DeviceBuffer` (used in many field declarations for master, gradient, momentum buffers)
- `backend/src/llm/src/training/transformer.rs`: references to `DeviceBuffer` in `test_gpu_sgd_does_not_corrupt_weights` test
- Any other file importing `device_buffer::DeviceBuffer`

### Shape of the fix
1. Rename the struct in `device_buffer.rs`: `pub struct Buffer { ... }`, all `impl Buffer { ... }`
2. Update `use` statements and field types in `gpu_workspace.rs`, `transformer.rs` tests
3. `DeviceBufferF16` is unaffected (it's a separate struct with a distinct name)

### Verification
- `cargo build -p llm` passes with zero warnings
- All existing tests pass (`cargo test -p llm`)
- No references to `device_buffer::DeviceBuffer` remain

## Comments
_No comments yet._
