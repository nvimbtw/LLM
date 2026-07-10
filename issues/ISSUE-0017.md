---
id: ISSUE-0017
title: Missing doc comments on pub items across backend (R9)
scope: backend
priority: P2
status: todo
rule: R9
location: backend/src/db/ backend/src/crypto/ backend/src/storage.rs backend/src/jobs.rs backend/src/training.rs backend/src/llm/src/training/
blocked-by: []
blocks: []
---

## Description
Nearly every `pub fn`, `pub struct`, and `pub enum` in `db/` (User, Chat, Message,
Model, Usage — ~30 pub items), `crypto/` (Claims, jwt, password, token_cipher — 6
pub items), `storage.rs` (9 pub fns), `jobs.rs` (JobStatus, JobState, Job,
JobRegistry — 4 items), `training.rs` (TrainConfig, DatasetStats, TokenizeEvent — 5
items), and several items in `llm/src/training/` (DeviceBuffer, DeviceBufferF16,
GpuWorkspace, LayerRow, ModelHeader, StepDims — ~15 items) lack `///` doc comments.
R9 requires every `pub` item to document its contract.

## Comments
_No comments yet._
