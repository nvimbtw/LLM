---
id: ISSUE-0063
title: "Rename training::gpu_workspace::GpuWorkspace to training::gpu_workspace::Workspace"
scope: backend/llm
priority: P2
status: verification
rule: R21
location: backend/src/llm/src/training/gpu_workspace.rs:9
blocked-by: []
blocks: [ISSUE-0060]
---

## Description

`training::gpu_workspace::GpuWorkspace` partially stutters — the module name `gpu_workspace` and the type name `GpuWorkspace` both carry the `gpu` prefix, so the path reads redundantly. Per **R21**, rename to `Workspace` so the path reads `training::gpu_workspace::Workspace` — the module already conveys "GPU-related," and within it, `Workspace` is unambiguous.

### Scope of change
- `backend/src/llm/src/training/gpu_workspace.rs`: rename `pub struct GpuWorkspace` → `pub struct Workspace`, update all `impl GpuWorkspace` blocks
- `backend/src/llm/src/training/transformer.rs`: all method signatures that take `&mut GpuWorkspace` or `&GpuWorkspace` (approx. 15 methods + helper functions `gpu_adam_update`, `gpu_clip_gradients`, `sync_weights_from_gpu`)
- `backend/src/llm/src/train.rs`: `let mut ws = GpuWorkspace::new(...)` calls and function parameter types
- `backend/src/llm/src/infer.rs`: (if referenced — check)

### Shape of the fix
1. Rename struct in `gpu_workspace.rs`: `pub struct Workspace`
2. `s/GpuWorkspace/Workspace/g` in `transformer.rs` (method signatures, test code)
3. Update references in `train.rs` (three `GpuWorkspace::new` calls + type annotations)
4. Update references in any other file importing `gpu_workspace::GpuWorkspace`

### Verification
- `cargo build -p llm` passes with zero warnings
- All existing tests pass (`cargo test -p llm`)
- No references to `gpu_workspace::GpuWorkspace` remain

## Comments
### Builder · 2026-07-14
Renamed `GpuWorkspace` -> `Workspace` across `gpu_workspace.rs`, `transformer.rs`, `train.rs`, tests. Build/clippy clean, 45/45 lib tests pass. Path now `training::gpu_workspace::Workspace`.
