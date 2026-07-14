---
id: ISSUE-0070
title: "Doc comments on Transformer save/load/bytes methods merely restate the method names"
scope: backend/llm
priority: P3
status: todo
rule: R9
location: backend/src/llm/src/training/transformer.rs:1028 backend/src/llm/src/training/transformer.rs:1034 backend/src/llm/src/training/transformer.rs:1039-1040 backend/src/llm/src/training/transformer.rs:1670 backend/src/llm/src/training/transformer.rs:1677
blocked-by: []
blocks: []
---

## Description

Five `pub fn` doc comments in `transformer.rs` merely restate what the method name already says — they add no contract information (units, invariants, panics, ownership) that a caller would not already infer from the name. Violates **R9 (document contracts, not names)** and **R19 (comments earn their place)**.

### Affected doc comments

| Line | Method | Current doc comment | Why it's a violation |
|------|--------|-------------------|---------------------|
| 1028 | `save_to_path` | `/// Serialize the model to \`path\` in the versioned \`GORK\` binary format.` | "Serialize the model to path" is what `save_to_path` means. The format note is discoverable from the constant `MODEL_MAGIC`/`MODEL_VERSION`. |
| 1034 | `save` | `/// Serialize the model to \`model.bin\` inside \`dir\`.` | Restates `save`. The filename is an implementation detail of this one-line delegation. |
| 1039-1040 | `load` | `/// Load a model from \`model.bin\` inside \`dir\`, accepting any supported format version. Errors on bad magic, truncation, or an invalid header.` | The first sentence restates `load`. The errors listed are standard IO errors any file reader produces. |
| 1670 | `to_bytes` | `/// Serialize the model to an in-memory buffer (same format as \`save\`).` | Restates `to_bytes`. The format reference is an implementation detail. |
| 1677 | `from_bytes` | `/// Deserialize a model from a byte buffer produced by \`to_bytes\`.` | Restates `from_bytes`. The round-trip reference is implied by the type signature. |

### Shape of the fix
Delete all five doc comments. The method names (`save_to_path`, `save`, `load`, `to_bytes`, `from_bytes`) are self-documenting with conventional Rust signatures (`Result<(), io::Error>` / `Result<Self, io::Error>`).

If the `GORK` format versioning and error conditions need documenting, move that information to a module-level comment above the serialization section (around line 1047 `// ── Model (de)serialization ──`), not on individual methods.

### Verification
- `cargo build -p llm` compiles clean
- No warnings about missing docs on these `pub fn` items (R7)
- Public API documentation for `Transfomer` (or `Model` after ISSUE-0061) is not degraded

## Comments
_No comments yet._
