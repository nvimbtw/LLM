---
id: ISSUE-0016
title: Unsafe blocks in device_buffer.rs and gpu_workspace.rs lack safety comments (R4)
scope: backend/llm
priority: P1
status: todo
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
_No comments yet._
