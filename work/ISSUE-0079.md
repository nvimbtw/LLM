---
id: ISSUE-0079
type: issue
title: "Extract (de)serialization into transformer/serialization.rs"
scope: backend/llm
priority: P2
status: verification
rule: R20
location: backend/src/llm/src/training/transformer/mod.rs:48-113 backend/src/llm/src/training/transformer/mod.rs:1055-1497 backend/src/llm/src/training/transformer/mod.rs:1669-end
blocked-by: [ISSUE-0078]
blocks: [ISSUE-0080]
---

## Description
### What
Move all model persistence — the byte-level read/write helpers and the
public (de)serialization API — out of `transformer/mod.rs` into a new
`transformer/serialization.rs`. This is the most self-contained concern (free
functions + a few methods over the struct's fields), so it goes first.

### What moves (by current line ranges in mod.rs)
- Header parsing: `read_model_header`, `validate_header_fields` (~48–113) and the
  header structs they use (`ModelHeader` / `ModelFileHeader` — move the ones that
  are serialization-only; leave any shared type in `mod.rs` and `use` it).
- Byte helpers + model IO: `write_2d`, `write_1d`, `write_model`, `read_u32`,
  `read_i64`, `read_u64`, `read_2d`, `read_1d`, `read_header`, `read_qkv_weights`,
  `read_block_weights`, `read_model` (~1055–1497).
- The (de)serialization methods on the model: `to_bytes`, `from_bytes`,
  `to_layer_rows`, `from_layer_rows` (~1671–end of the second `impl`), plus the
  thin `save`, `save_to_path`, `load` wrappers (~1029–1054) if they are pure IO —
  keep them wherever reads cleanest, but co-locate them with `write_model`/
  `read_model` they delegate to.

### Shape of the change
- New file `transformer/serialization.rs`; add `mod serialization;` (private) to
  `transformer/mod.rs`, and `use` back into `mod.rs` any item the rest of the
  module still calls (e.g. `load`/`save` if referenced from `train.rs` via the
  `Model` impl — keep the public path stable).
- Methods move as an additional `impl Model { … }` block in the new file.
- Adjust visibility: free helpers stay private to `serialization`; anything called
  from outside becomes `pub(super)` or keeps its existing `pub`. Do not widen
  beyond current call sites.

### Verification
- `cargo build` + `cargo clippy` clean; the public API path (`Model::load`,
  `Model::save`, `Model::to_bytes`, …) is unchanged so `train.rs`/`infer.rs`/tests
  compile without edits.
- Full GPU suite passes `--test-threads=1`; serialization round-trip tests
  (`test_to_bytes_from_bytes_roundtrip`, `test_to_layer_rows_*`) still pass.

## Comments
### Auditor · 2026-07-14
Serialization first because it is the cleanest cut — mostly free functions with a
known field layout and its own round-trip tests to prove the move was behaviour-
preserving.
