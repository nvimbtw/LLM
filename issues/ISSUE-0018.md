---
id: ISSUE-0018
title: from_layer_rows is 168 lines with 4 inline closures (R1, R13)
scope: backend/llm
priority: P2
status: todo
rule: R1, R13
location: backend/src/llm/src/training/transformer.rs:1668-1837
blocked-by: []
blocks: []
---

## Description
`from_layer_rows` spans 168 lines of body and defines 4 inline closures (`get_2d`,
`get_1d`, `_get_3d_heads` — itself ~60 lines — `read_qkv`) instead of extracting
them as named methods. The `_get_3d_heads` closure has three distinct code paths
(per-head rows, single 3D blob, fallback). R1 limits functions to ~60 lines; R13
says inline comment-explained blocks should be extracted functions.

## Comments
_No comments yet._
