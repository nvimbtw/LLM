---
id: ISSUE-0005
type: issue
title: Split load and from_bytes — each ~150 lines (R1)
scope: backend/llm
priority: P2
status: done
rule: R1
location: backend/src/llm/src/training/transformer.rs:909-1057,1508-1650
blocked-by: [ISSUE-0003]
blocks: []
---

## Description
`load` (148 lines) and `from_bytes` (142 lines) contain the same versioned
field-reading loop repeated for each weight tensor, well over the R1 limit. After
the ISSUE-0003 de-duplication, split the header-parse and per-layer-read stages
into named functions.

## Comments
### Verifier · migrated · pass
`load` is now 4 lines (delegates to `read_model`), `from_bytes` is 2 lines. Both
well under 60.
