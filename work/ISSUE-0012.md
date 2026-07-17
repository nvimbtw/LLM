---
id: ISSUE-0012
type: issue
title: No bounds validation on model file header fields (R10)
scope: backend/llm
priority: P1
status: done
rule: R10
location: backend/src/llm/src/training/transformer.rs:24-47,909-1057,1508-1650
blocked-by: []
blocks: []
---

## Description
`read_model_header`, `load`, and `from_bytes` read dimensions, heads, layers, etc.
from a model file (external input) and immediately use them in arithmetic without
validating they are sane positive values. If `heads` is 0, `dimensions / heads`
(line 961 / from_bytes equivalent) panics with division by zero. Validate at the
boundary: reject any header field ≤ 0 before proceeding.

## Comments
### Verifier · migrated · pass
Confirmed `validate_header_fields` (lines 66-97) checks all 6 header fields are
positive and that `dimensions % heads == 0`. Called from both `read_header` and
`read_model_header`.
