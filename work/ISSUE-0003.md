---
id: ISSUE-0003
type: issue
title: De-duplicate Transformer serialization read/write logic
scope: backend/llm
priority: P2
status: done
rule: R6
location: backend/src/llm/src/training/transformer.rs:850-1057,1438-1650
blocked-by: []
blocks: []
---

## Description
`save_to_path`/`to_bytes` and `load`/`from_bytes` carry near-identical
serialization logic (the same field order, the same 2d/1d read/write helpers,
the same versioned per-layer layout) duplicated across `File` and `Cursor`/`Vec`
targets. They must change together on every format bump — a classic R6 hazard.
Factor the layout into one place parameterized over the reader/writer (e.g. via
`std::io::Read`/`Write` trait objects) so there's a single definition of the file
format.

## Comments
### Verifier · migrated · pass
Confirmed `write_model`/`read_model`/`read_header`/`read_qkv_weights`/
`read_block_weights`/`read_2d`/`read_1d` shared by file and bytes paths.
`save_to_path`/`to_bytes`/`load`/`from_bytes` all delegate. Verified no remaining
duplicated serialization code.
