---
id: ISSUE-0045
type: issue
title: "Params rail — config as an inline-editable hairline definition list (refactor ConfigForm)"
scope: frontend
priority: P2
status: verification
rule: none
location: frontend/components/training/config-form.tsx frontend/components/training/params-rail.tsx
blocked-by: [ISSUE-0044]
blocks: [ISSUE-0048]
---

## Description

Part of the `/training` redesign epic (EPIC-0040). Restyles the configuration form
into the left-rail **params panel** of the editorial spread: the run's identity and
knobs read as a hairline **definition list** (label ⟷ value rows separated by 1px
rules), not a boxed 2–3 column grid of inputs.

### What's there now

`components/training/config-form.tsx` renders a `grid grid-cols-2 sm:grid-cols-3` of
boxed `Input`s with a mode toggle, base-model select, derived-value line, error
list, and submit button (`config-form.tsx:156-259`). The *logic* is good and must be
preserved — mode-aware field sets (`ARCH_FIELDS`/`RUN_FIELDS`), the
`validate()`/`validateName()` rules mirroring the backend, the SFT default-LR nudge,
the derived head-size / FF-hidden readout. This issue changes the **presentation and
placement**, not the validation.

### The editorial rail

Extract the presentation into `components/training/params-rail.tsx` (the form logic
can stay in a hook or the same file — keep one component per file per conventions):

- **Definition-list rows.** Each field is a row: a muted label on the left, the
  editable value right-aligned in `tabular-nums`, a hairline `border-b` between rows.
  The value is edit-in-place — a borderless `Input` that looks like text until
  focused (bottom-border or subtle bg on focus), so at a glance it reads as an
  article's spec sheet, not a form. Keep them keyboard-navigable and labelled.
- **Model name** as the rail's headline (larger, `font-heading`), since it's the
  run's identity — echoes the masthead.
- **Mode** stays a compact segmented toggle (Pretrain / Fine-tune) at the top of the
  rail; switching it swaps the visible field set exactly as today (`numFields`).
- **Derived values** (head size, FF hidden) become quiet caption rows in the list,
  not a separate boxed line.
- **Errors** render inline and quiet — a muted `text-destructive` line under the
  offending row or a small cluster above the submit, not a bulleted block that
  shoves the layout. Submit stays disabled while `errors.length > 0`.
- **Start button** anchored at the rail's foot, full-width, labelled per mode
  (Start training / Start fine-tune) as today.
- A **placeholder slot** at the top of the rail for the preset controls (picker +
  save) — that UI is child 0048; leave a clearly-marked mount point so 0048 drops in
  without re-laying-out the rail.

### Rules

- **Preserve every validation and the derived-value math** verbatim — reuse
  `validate`, `validateName`, `isPowerOfTwo`, `ARCH_FIELDS`, `RUN_FIELDS`,
  `SFT_DEFAULT_LR`, the `effectiveCfg`/`baseModelId` derivation. Don't rewrite the
  logic while restyling.
- **No boxes.** Hairline rules via the `border` token + `cn()`; no `Card`. Values in
  `tabular-nums`. Reuse shadcn `Input`/`Button`; add a variant/prop if a borderless
  input is needed rather than hand-rolling one (frontend convention: extend the
  primitive, don't replace it).
- The rail lives in the shell's left-rail slot (0044); it must not reflow when the
  run starts — while a run is active the fields go read-only/`disabled` (as today via
  the `disabled` prop) but stay in place, so the rail doubles as the run's config
  recap (replacing the old `configSummary` collapse behaviour).

### Definition of done

- The config renders as a hairline definition list in the left rail; all fields,
  the mode toggle, base-model select, derived readouts, validation errors, and the
  start button work exactly as before.
- Disabled (mid-run) state keeps the rail in place, read-only, no reflow.
- `bun run typecheck` + `bun run lint` clean.

## Comments
### Auditor · 2026-07-12
Child of EPIC-0040, blocked by the shell (0044) which owns the rail slot. Scoped
to presentation + placement with the validation logic held constant, so the risk is
purely visual. Leaves a mount point for the preset controls (0048).

### Builder · 2026-07-12
Added `components/training/params-rail.tsx` — the config as an inline-editable hairline definition list (borderless inputs, `border-b` rows, right-aligned `tabular-nums`), model name as the rail headline, compact mode toggle, SFT base-model select row, derived head-size/FF-hidden caption rows. All validation ported verbatim (`validate`/`validateName`/`isPowerOfTwo`, ARCH/RUN field split, `SFT_DEFAULT_LR` nudge, `effectiveCfg`/`baseModelId` derivation). Config state lifted to `page.tsx` so the continuation flow can seed it; the rail edits via `onChange`. Mounts `PresetControls` at the top.
