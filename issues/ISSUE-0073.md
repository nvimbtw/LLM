---
id: ISSUE-0073
title: "R19 section banners in training.rs, device_buffer.rs, and infer.rs"
scope: backend
priority: P3
status: todo
rule: R19
location: backend/src/training.rs:358 backend/src/training.rs:509 backend/src/llm/src/training/device_buffer.rs:102 backend/src/llm/src/infer.rs:263 backend/src/llm/src/infer.rs:318 backend/src/llm/src/infer.rs:380
blocked-by: []
blocks: []
---

## Description

Section-banner comments in three files narrate *what* the following code does without adding information the code structure already provides. Violates **R19 (comments earn their place)**.

### Locations

| File | Line | Comment | Why it's noise |
|------|------|---------|---------------|
| `training.rs` | 358 | `// ── Supervised fine-tuning (SFT) dataset ──` | Precedes a section of the file already organized by function definitions; the comment adds no information the function names don't already provide. |
| `training.rs` | 509 | `// ── Synthetic SFT data generator (ISSUE-0036) ──` | Precedes `mod synthetic`, which is already self-documenting as its own module. The comment is redundant. |
| `device_buffer.rs` | 102 | `// ─── f16 GPU buffer ───` | Precedes the `DeviceBufferF16` struct definition. The struct name itself (`DeviceBufferF16`) and its position in a file called `device_buffer.rs` already convey this. |
| `infer.rs` | 263 | `// ── marker_from_str ──` | Test section banner. The tests below are already named. |
| `infer.rs` | 318 | `// ── sample_token ──` | Test section banner. |
| `infer.rs` | 380 | `// ── find_markers ──` | Test section banner. |

### Shape of the fix
Delete each banner comment. The code structure (function names, module boundaries) already conveys the same information.

For the test section banners at `infer.rs:263/318/380`, the tests would be clearer if grouped inside their own `#[cfg(test)] mod tests { ... }` with a descriptive `mod` name instead of a comment banner. Consider moving them into the existing test module at line 259.

### Verification
- No section-banner comments remain at the listed locations
- `cargo build` compiles clean
- All tests pass

## Comments
_No comments yet._
