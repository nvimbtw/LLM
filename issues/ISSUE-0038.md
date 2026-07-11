---
id: ISSUE-0038
title: Frontend — fine-tune (SFT) mode toggle and base-model picker
scope: frontend
priority: P3
status: done
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

### VCS · 2026-07-10
Committed as `d5b9a57` on branch `dev`.

### Builder · 2026-07-11
Surfaced the two training modes in the training UI.

**What changed:**
- **`lib/training-socket.ts`:** added `type TrainMode = "pretrain" | "sft"` and,
  on `TrainConfig`, `mode: TrainMode` + `base_model_id?: string` — the exact field
  names/casing the backend (`TrainConfig` in ISSUE-0037) deserializes.
- **`hooks/use-training.ts`:** `DEFAULT_CONFIG` gains `mode: "pretrain"` so the
  default request is byte-identical to before plus an explicit (backend-default)
  mode. `base_model_id` is left unset for pretrain.
- **`components/training/config-form.tsx`:**
  - **Mode toggle** at the top (two `Button`s as a segmented control; the active
    mode is `variant="default"`, the other `"ghost"`). Switching to SFT also drops
    the learning rate to `SFT_DEFAULT_LR` (1e-4) — but only if the user hadn't
    moved it off the pretrain default, and it restores on switching back.
  - **Field split:** `ARCH_FIELDS` (dimensions, ctx_window, heads, layers,
    ff_multiplier, max_vocab, num_books, num_synthetic) render only in pretrain;
    `RUN_FIELDS` (epochs, lr) render in both. So SFT hides everything the base
    model dictates and keeps only what a fine-tune actually chooses. The
    head-size/FF hint is pretrain-only too.
  - **Base-model picker** (SFT only): a native `<select>` (there is no shadcn
    `select` primitive, and the form already uses native inputs) populated from
    the existing `useModels()` hook — reusing the app's models fetch rather than a
    new data path, as the issue asked. Styled to match the `Input` primitive.
  - **Validation:** `validate` branches on mode. SFT validates only name, a
    selected base model, and positive epochs/lr — the power-of-two / divisibility
    checks (which mirror the backend `validate_config`) don't fire, matching that
    those params are ignored server-side for SFT. Pretrain validation is unchanged.
  - **Derived base id (no setState-in-effect):** rather than an effect that
    preselects the first model (which trips the repo's
    `react-hooks/set-state-in-effect` lint), the effective base id is *derived* at
    render — `cfg.base_model_id || models[0]?.id` — and folded into an
    `effectiveCfg` used for both validation and submit. The `<select>` stays
    controlled off that derived value.
  - Submit sends `effectiveCfg`; in SFT that's `{ mode:"sft", base_model_id,
    model_name, epochs, lr, … }`. The inherited architecture fields still ride
    along at their defaults (valid, and ignored by the backend for SFT), so no
    request-shape surprise.

**Back-compat:** pretrain mode renders the identical field set and submits the
same shape as before (now with an explicit `mode:"pretrain"`, the backend
default) — so existing behaviour is unchanged.

**Verified:**
- `bun run typecheck` — clean.
- `bun run build` — succeeds (all 13 routes generated).
- `bun run lint` — my three changed files (`config-form.tsx`, `training-socket.ts`,
  `use-training.ts`) are clean. Note: `bun run lint` reports 13 pre-existing
  errors in files I did **not** touch (`use-chat.ts`, `use-models.ts`,
  `use-usage.ts`, `chat-composer.tsx`, …), all the same
  `react-hooks/set-state-in-effect` rule — a baseline condition on `dev`, not
  introduced here (and the reason I deliberately avoided that pattern in the new
   code). Flagging for the Auditor as a separate pre-existing cleanup if wanted.

### Verifier · 2026-07-11 · pass

**Commands run in `frontend/`:**
- `bun run typecheck` — clean (zero errors).
- `bun run build` — succeeds, all 13 routes generated cleanly.
- `bun run lint` — reports 13 problems (12 errors / 1 warning), all in **untouched** files: `chat-composer.tsx` (3× `react-hooks/refs`), `chats-provider.tsx`, `flux-throbber.tsx`, `model-edit-sheet.tsx`, `use-auth.ts`, `use-chat.ts`, `use-mobile.ts`, `use-model.ts`, `use-models.ts`, `use-usage.ts`. None of the three changed files appear — confirming the Builder introduced no new lint issues.

**Code review — all three files match the issue spec:**

- `lib/training-socket.ts`: `TrainMode` type and `TrainConfig.mode`/`base_model_id?` added with correct field names matching backend contract.
- `hooks/use-training.ts`: `DEFAULT_CONFIG` gains `mode: "pretrain"` — back-compat byte-identical default.
- `components/training/config-form.tsx`:
  - Mode toggle (segmented control with Buttons) at the top ✅
  - SFT mode hides architecture fields (ARCH_FIELDS), shows only RUN_FIELDS + base-model picker ✅
  - Base-model picker populated from `useModels()` hook — reuses existing fetch path ✅
  - `validate()` branches on mode; SFT skips power-of-two/divisibility checks, requires base model ✅
  - Derived `baseModelId` (`cfg.base_model_id || models[0]?.id`) — no setState-in-effect ✅
  - Submit sends `effectiveCfg` with resolved `base_model_id` in SFT mode ✅
  - Pretrain mode field set + submission shape unchanged ✅

**Rules check (R14–R18):**
- R14: `"use client"` on files that need it; `training-socket.ts` correctly omits it (pure utility).
- R15: All API calls go through `lib/api.ts` (via `useModels` / `startTraining` / `attachTraining` / `listJobs`).
- R16: Uses shadcn `Button` and `Input`; native `<select>` where no shadcn primitive exists (consistent with existing form pattern).
- R17: One component/hook per file.
- R18: No `any` — all types explicit (uses `Record<string, unknown>`, proper `interface`/`type`).

No rule violations. The fix is complete, back-compatible, and adds zero new lint issues.
