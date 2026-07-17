---
id: ISSUE-0049
type: issue
title: "Pretrain→SFT continuation — stepper transition + Continue-to-SFT seeding the base from the finished pretrain"
scope: frontend
priority: P2
status: verification
rule: none
location: frontend/lib/training-socket.ts frontend/hooks/use-training.ts frontend/components/training/training-shell.tsx frontend/app/training/page.tsx
blocked-by: [ISSUE-0043, ISSUE-0044]
blocks: []
---

## Description

Part of the `/training` redesign epic (EPIC-0040). Delivers the flow the user
asked for: **"after normal training there should be a button that advances this to
SFT."** When a pretrain finishes, the phase stepper advances and a **"Continue to
SFT"** action starts a fine-tune whose base is the model that was just trained — no
hunting for it in the base-model dropdown.

### Consume the new backend fields (from 0043)

Child 0043 makes the `done`/`snapshot` frames carry `id_model` + `model_name`.
Thread them through the frontend:

- Extend `TrainingHandlers.onDone` and the `TrainingSnapshot` type in
  `lib/training-socket.ts` with `id_model?: string` / `model_name?: string`.
- In `hooks/use-training.ts`, capture them on `onDone`/`onSnapshot` into new hook
  state (e.g. `finishedModel: { id, name } | null`), so the shell knows *which*
  model a completed pretrain produced. They're absent on cancel/fail → stays null.

### The stepper transition (in the shell, 0044)

- On a **successful pretrain** (`status === "done"` and the finished run's
  `mode === "pretrain"` with a `finishedModel`), the masthead phase stepper marks
  Pretrain done and lights the SFT step as available, and a **"Continue to SFT"**
  button appears (masthead primary action).
- Clicking it seeds a new SFT config: `mode: "sft"`, `base_model_id:
  finishedModel.id`, a sensible derived `model_name` (e.g. `${finishedModel.name}-sft`),
  and the SFT-appropriate LR/epoch defaults already encoded in the config form
  (`SFT_DEFAULT_LR` etc.). The rail switches to SFT mode showing the base preselected
  and read-only-inheriting architecture — exactly the existing SFT form (0045), just
  pre-filled. The user can tweak epochs/LR, then Start.
- The SFT run then proceeds through the same shell (dataset build → train),
  telemetry populating the same slots. **No pop, no page jump** — the stepper's
  advance and the rail's mode switch happen in place (the epic's core rule).

### Preserve or archive phase-1 telemetry

Starting the SFT run resets the live telemetry state (`start()` clears loss/charts —
`use-training.ts:191-201`). That's acceptable for v1 — the pretrain's model is
already registered and its final numbers were shown. Do **not** try to keep both
runs' charts side by side (out of scope). But: the stepper must still read "Pretrain
✓" after the switch, so keep enough completed-phase state (the `finishedModel` +
a `pretrainDone` flag) to render the stepper correctly across the transition — don't
let `start()` wipe the fact that pretrain succeeded.

### Reattach case

A pretrain that finished while the tab was closed re-attaches via `snapshot` with the
new id/name fields (0043) — so opening `/training` on a just-finished pretrain also
shows "Continue to SFT". Make sure the `onSnapshot` path populates `finishedModel`
the same way `onDone` does.

### Rules

- Reuse the existing `useTraining.start(cfg)` to launch the SFT run — the seeding is
  just building the right `TrainConfig` and handing it to `start`. No new socket
  path.
- The manual base-model picker (from the SFT form) stays available for the "I want
  to fine-tune some *other* model" case — Continue-to-SFT is a shortcut, not the
  only route into SFT.

### Definition of done

- Finishing a pretrain surfaces "Continue to SFT"; clicking it starts an SFT run
  whose base is the just-trained model, with no manual base pick.
- The phase stepper reads Pretrain ✓ → SFT active across the transition, in place,
  with no card pop-in / page-height jump.
- A reattached, just-finished pretrain also offers Continue-to-SFT.
- `bun run typecheck` + `bun run lint` clean.

## Comments
### Auditor · 2026-07-12
Child of EPIC-0040, blocked by the backend id/name fields (0043) and the shell +
stepper (0044). This is the "advance to SFT" flow from the user brief and the payoff
of making pretrain/SFT lateral. Scoped to v1 (reset telemetry on the SFT start, keep
the completed-pretrain state for the stepper) to stay bounded.

### Builder · 2026-07-12
Extended `lib/training-socket.ts` (`TrainingSnapshot`/`onDone` gain `id_model`/`model_name`; `FinishedModel`) and `use-training.ts` (`runConfig`, `finishedModel`, sticky `pretrainDone`; captured on `onDone`+`onSnapshot` so a reattached finished pretrain also qualifies; `start()` keeps `pretrainDone` for an SFT continuation but clears it for a fresh pretrain). `page.tsx` shows a masthead **Continue to SFT** action when a pretrain finished, seeding `mode:sft` + `base_model_id` + `<name>-sft` + SFT LR and switching the rail in place. Stepper reads Pretrain ✓ → SFT active across the switch. Live-run + continuation behaviour (pop-free during a real run) needs the GPU backend and is left for the user to confirm visually.
