---
id: ISSUE-0038
title: Frontend — fine-tune (SFT) mode toggle and base-model picker
scope: frontend
priority: P3
status: todo
rule: none
location: frontend/components/training/config-form.tsx:12-147 frontend/hooks/use-training.ts:15-27 frontend/lib/training-socket.ts
blocked-by: [ISSUE-0037]
blocks: []
---

## Description
### What
Surface the two training modes in the training UI: a **Pretrain / Fine-tune**
toggle on the config form, and — when Fine-tune is selected — a **base-model
picker** (which existing model to fine-tune) plus the SFT-relevant knobs, wired
into the request `TrainConfig` (ISSUE-0037's `mode` + `base_model_id`).

### Why
ISSUE-0037 makes SFT launchable over the API, but the only way to start a run
today is `config-form.tsx`, which assumes pretraining (it always sends the full
architecture config — dimensions/heads/layers/etc.). A fine-tune doesn't choose a
new architecture; it inherits the base model's, so the form needs to adapt.

### Shape of the fix
- **`TrainConfig` type (`lib/training-socket.ts` + `DEFAULT_CONFIG` in
  `hooks/use-training.ts:15`):** add `mode: "pretrain" | "sft"` (default
  `"pretrain"`) and `base_model_id?: string`, matching the backend field names.
- **`config-form.tsx`:** add a mode toggle at the top. In `sft` mode:
  - Hide/disable the architecture fields that come from the base model
    (dimensions, ctx_window, heads, layers, ff_multiplier, max_vocab) — a
    fine-tune can't change them. Keep `epochs`, `lr` (default `lr` lower for SFT),
    and `model_name` (the output model's display name).
  - Show a base-model `<select>` populated from the existing models list. The
    backend already exposes models (see `routes/model.rs` / the model list the app
    uses elsewhere) — reuse that fetch rather than adding a new client convention.
  - Adjust `validateConfig` so the architecture-field "must be positive" checks
    don't fire in SFT mode, and add "a base model must be selected" for SFT.
- Match the existing form styling/patterns (the `NUM_FIELDS` map, the label/Input
  layout) — don't introduce a new form abstraction. The train panel/charts
  (`train-panel.tsx`) need no changes; they already render whatever loss stream
  the run emits.

### Verification
- `bun run lint` / typecheck / build clean in `frontend/`.
- Pretrain mode is unchanged (same fields, same request shape) — this is the
  back-compat guard.
- Selecting Fine-tune shows the base-model picker, hides architecture fields, and
  submits `{ mode: "sft", base_model_id, model_name, epochs, lr }`.

## Comments
### Auditor · 2026-07-10
The last child of the SFT epic — the only frontend piece. P3 because the feature
is fully usable via the API once 0037 lands; this is the ergonomic front door.
Blocked-by 0037 (needs the `mode`/`base_model_id` contract and the branching
route). Main care point: keep pretrain mode's request shape identical so existing
behaviour doesn't regress, and reuse the existing models-list fetch rather than
inventing a new data path (frontend conventions in the epic / existing hooks).
