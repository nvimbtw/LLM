---
id: ISSUE-0071
title: "R19 narrative comments and section banners in transformer.rs should be removed or extracted"
scope: backend/llm
priority: P2
status: todo
rule: R19
location: backend/src/llm/src/training/transformer.rs:482 backend/src/llm/src/training/transformer.rs:518-636 backend/src/llm/src/training/transformer.rs:640-699 backend/src/llm/src/training/transformer.rs:805-900 backend/src/llm/src/training/transformer.rs:1047
blocked-by: []
blocks: []
---

## Description

Multiple comments in `transformer.rs` violate **R19 (comments earn their place)** — they narrate *what* the next code does rather than explaining *why* it's done. These fall into two categories:

### 1. Section banners that should be module boundaries

- **Line 482**: `// ── train_step stage helpers ──` — organizational banner separating the train-step stage helpers. These helpers are already grouped as `impl Transformer` methods within the same file; the banner adds nothing the code structure doesn't already provide.
- **Line 1047**: `// ── Model (de)serialization ──` — separates free functions for read/write from the `impl Transformer` block. Indicates this portion of the file should be its own module (e.g., `training/serialization.rs` or `training/persistence.rs`).

### 2. Per-CUDA-kernel narrative comments (lines ~518–900)

Every stage helper method in the train-step pipeline carries a comment block that describes *what* the following `unsafe` block does — e.g.:
- **Line 530**: `// Save layer input (for LN1 backward + residual).` before a `cuda_memcpy_d2d`
- **Line 540**: `// Layer norm 1 (pre-attention), then cache its output (= attention input).`
- **Line 559**: `// Batched multi-head self-attention forward.` before `cuda_attention_forward_batched`
- **Line 573**: `// Concat heads: interleave by token on GPU.` before the concat-heads code
- **Line 598**: `// Save concat output (for wo backward), then project through wo.`
- **Line 616-617**: `// Residual add: x + attn_out (restore x from cache), save the post-attention residual...`
- **Line 683**: `// Residual add: post_attn + ff_out.`
- **Line 701**: `// Project the final hidden state to logits in d_logits.`
- **Line 763-764**: `// No trained positions in this chunk: produce no loss and a zero gradient...`
- **Line 805-806**: `// Backprop through the output projection...`
- **Line 844**: `// Feed-forward backward, accumulate its weight/bias gradients.`
- **Line 869**: `// Layer norm 2 backward, accumulate its gamma/beta gradients.`
- **Line 888**: `// Residual: d_dx + LN2 backprop → b_sd[2].`

Each of these comments narrates what the immediately following `unsafe` CUDA kernel call(s) do. The kernel function names (`cuda_attention_forward_batched`, `cuda_layer_norm`, `cuda_matmul`, `cuda_add_inplace_f16`, etc.) already state *what*; the comment is noise per R19.

The `SAFETY: see contract.` line within each block IS justified (R4 requires safety justification) — keep only the `// SAFETY: see contract.` part and delete the narrative prefix.

### Shape of the fix

1. **For section banners at lines 482 and 1047**: Delete them. If the serialization code at line 1047+ is large enough to warrant organization, extract it into its own module (`training/serialization.rs` or similar) rather than captioning it.

2. **For per-kernel narrative comments (lines ~518-900)**: In each `unsafe { ... }` block, keep only `// SAFETY: see contract.` and remove the preceding "what this does" prose. The safety contract at lines 484-490 already documents the shared invariant; the individual kernel names (`cuda_attention_forward_batched`, etc.) state the operation.

### Verification
- No comments remain that merely narrate a kernel call the name already describes
- Section banners no longer exist in `transformer.rs`; if extraction was chosen instead, the new module compiles
- All `SAFETY:` annotations are preserved (R4 compliance)
- `cargo build -p llm` compiles clean

## Comments
_No comments yet._
