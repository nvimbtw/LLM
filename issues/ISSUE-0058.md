---
id: ISSUE-0058
title: "Mode-aware default learning rates (pretrain ~1e-4, SFT ~1e-5) now that the LR cap is lifted"
scope: backend
priority: P2
status: todo
rule: none
location: backend/src/llm/src/train.rs backend/src/training.rs frontend/hooks/use-training.ts
blocked-by: [ISSUE-0057]
blocks: []
---

## Description
### What
Raise the **default** learning rate for pretraining now that 0057 has proven the
f32-master path trains stably above the old ~1e-5 cap, and set a distinct, lower
SFT default (fine-tuning wants a gentler LR than from-scratch pretrain). This is
the ergonomic payoff of the epic — the fix is worthless to users if the default
still targets the old survivable-but-undertrained LR.

### Shape of the change
- Use the LR ceiling **observed in 0057** as the evidence for the number; target
  pretrain default **~1e-4** and SFT default **~1e-5** (adjust to whatever 0057
  actually showed stable — do not exceed the proven-stable value).
- Set the defaults where they live today: the pretrain/SFT entry points in
  `train.rs` / `backend/src/training.rs`, and the frontend `DEFAULT_CONFIG` /
  `SFT_DEFAULT_LR` in `frontend/hooks/use-training.ts` (ISSUE-0038 already added
  an SFT-specific default LR knob — reuse it, don't add a second convention).
- Keep the two modes' defaults independent so switching Pretrain↔Fine-tune sets
  the right LR (the config form already restores per-mode defaults on toggle).

### Out of scope
- No new UI, no validation-rule changes — only the default *values*.
- The smaller-vocab default is a **separate** issue (ISSUE-0059); do not fold it
  in here.

### Verification
- Backend builds/tests clean; frontend `bun run typecheck` + `build` clean.
- Starting a pretrain with no LR override now uses the raised default; starting an
  SFT run uses the SFT default; the numbers match 0057's proven-stable ceiling.

## Comments
### Auditor · 2026-07-13
Last child in the chain — pure defaults, gated on 0057 so the raised number is
backed by an actual stable run rather than optimism. Kept the vocab default out
(ISSUE-0059) so this stays a single-concern change.
