---
id: ISSUE-0060
title: "Epic: Non-stuttering type names in the llm::training module"
scope: backend/llm
priority: P2
status: todo
rule: R21
location: backend/src/llm/src/training/transformer.rs:114 backend/src/llm/src/training/device_buffer.rs:11 backend/src/llm/src/training/gpu_workspace.rs:9
blocked-by: [ISSUE-0061, ISSUE-0062, ISSUE-0063]
blocks: []
---

## Description

epic — no code lives here directly. Three types in `llm::training/` stutter against their module name, violating **R21 (names don't stutter)**:

| Child | Current path | Violation | Proposed name |
|-------|-------------|-----------|---------------|
| ISSUE-0061 | `training::transformer::Transformer` | `transformer::Transformer` repeats the module name | `transformer::Model` |
| ISSUE-0062 | `training::device_buffer::DeviceBuffer` | `device_buffer::DeviceBuffer` repeats the module name | `device_buffer::Buffer` |
| ISSUE-0063 | `training::gpu_workspace::GpuWorkspace` | `gpu_workspace::GpuWorkspace` repeats the `gpu` prefix | `gpu_workspace::Workspace` |

### Definition of done
All three children are verified `done`. `cargo build` passes with zero warnings. No stuttering type names remain in `backend/src/llm/src/training/`.

## Comments
_No comments yet._
