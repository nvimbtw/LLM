---
id: ISSUE-0044
title: "Editorial layout shell — masthead + phase stepper + two-column spread; retire the collapsible Section cards"
scope: frontend
priority: P2
status: verification
rule: none
location: frontend/app/training/page.tsx frontend/components/training/section.tsx frontend/components/training/training-shell.tsx
blocked-by: []
blocks: [ISSUE-0045, ISSUE-0046, ISSUE-0047, ISSUE-0049]
---

## Description

Part of the `/training` redesign epic (ISSUE-0040). This is the **structural
centerpiece**: it replaces the vertical stack of pop-in/pop-out `Card`s with the
**editorial spread** — a wide, always-present two-column layout — and establishes
the one rule the rest of the frontend children build on: *reserve the slot, fill it
in, never pop*.

### What's wrong today

`app/training/page.tsx` conditionally **mounts** the tokenize/train sections as the
run progresses (`page.tsx:76`, `:100`), and `components/training/section.tsx` is a
collapsible `Card` that auto-expands on going active and collapses when done
(`section.tsx:41-51`). Together they make the page grow, jump, and reflow. Also the
page is capped `max-w-4xl` (`page.tsx:48`) and single-column, wasting a wide screen.

### The shell

A new `components/training/training-shell.tsx` owns the frame; `page.tsx` becomes a
thin composition of it + the `useTraining` hook. Layout (matching the epic mock):

```
┌────────────────────────────────────────────────────────────┐
│  masthead:  model name · status dot · PHASE STEPPER · action │
├──────────────────┬─────────────────────────────────────────┤
│  left rail       │  main column                             │
│  (params slot)   │  stat-run slot                           │
│  ~320–360px      │  hero-figure slot                        │
│  fixed           │  secondary-figures slot                  │
│                  │  appendix slot (collapsed)               │
└──────────────────┴─────────────────────────────────────────┘
```

- **Widen it.** Drop `max-w-4xl`; use the available width (a generous `max-w-[…]` or
  full-bleed with padding) so a wide monitor is actually used. Left rail fixed width,
  main column fluid. Collapse to a single stacked column below `lg` (mobile keeps the
  order: params → stats → charts → appendix).
- **Named slots, always rendered.** The shell renders `children`/props into fixed
  regions that exist from first paint. This issue delivers the shell + the **phase
  stepper + masthead**, and wires the *existing* panels into the slots as-is (a
  straight port) so the page still works end-to-end. Restyling the params rail
  (0045), the stat run + figures (0046), and the appendix (0047) are the follow-ups
  that fill each slot properly. Until then, a slot with no data shows a quiet
  hairline placeholder with a muted caption — **never an empty/absent region**.

### The phase stepper (masthead)

A horizontal two-step indicator — **Pretrain › SFT** — that makes the two activities
lateral instead of stacked. It reflects the current run's `mode` and status:
- Pretrain active / done / SFT active / SFT done states, rendered editorially
  (hairline separators, a filled dot or rule for the active/done step, muted for the
  inactive one). No boxes.
- The stepper is presentational here; the *transition* behaviour ("Continue to SFT"
  advancing it) is child 0049. This issue just renders the correct state from
  `t.status` + the config's `mode`.

### The "never pop" rule (inherited by all frontend children)

- Regions are **always mounted**; they transition by *populating*, not by
  mounting/unmounting. Between phases, telemetry that isn't live yet shows a
  placeholder in its reserved slot.
- **Retire `section.tsx`** (the collapsible card). Its throbber-on-active idea can
  survive as a small inline status affordance in the masthead if useful, but the
  card chrome and auto-collapse behaviour go away. Grep for other importers before
  deleting; it's currently only used by `page.tsx`.
- No layout-height jump as status changes: the hero-figure slot reserves its height
  (the loss chart is already a fixed `h-64` — `train-panel.tsx:182`); keep that
  reserved even before the first loss point (today it already shows a dashed "waiting"
  box, which is the right idea — generalize it to every slot).

### Definition of done

- The page renders the two-column spread on a wide viewport with the existing
  tokenize/train/config content ported into the slots, functionally unchanged.
- Starting and finishing a (tiny) run causes **no card pop-in and no page-height
  jump** — regions fill in place.
- The loss chart is visible without scrolling on a 1440px window while training.
- `section.tsx` is removed (or reduced to the masthead status affordance) with no
  dangling imports. `bun run typecheck` + `bun run lint` clean.

### Notes for the Builder

- Keep using the `useTraining` hook and its state as-is — this is a *presentation*
  refactor, not a data-flow change. Don't move domain state out of the hook
  (frontend conventions: state lives in hooks).
- Use `cn()` + Tailwind + shadcn tokens; no new CSS files. Hairline rules =
  `border`/`divide` with the existing `border` token, not custom colors.

## Comments
### Auditor · 2026-07-12
Child of ISSUE-0040 and the linchpin of the frontend work — it establishes the
spread + the "never pop" rule and unblocks 0045/0046/0047/0049. Deliberately scoped
as a *straight port into slots* so it's shippable and verifiable on its own before
the per-slot restyles land.

### Builder · 2026-07-12
Added `components/training/training-shell.tsx` — masthead (identity · status dot · Pretrain›SFT phase stepper · primary-action slot) over a wide `lg:grid-cols-[340px_minmax(0,1fr)]` spread (`max-w-[1400px]`, sticky rail), collapsing to one column below `lg`. `page.tsx` is now a thin composition of the shell + `useTraining`. Retired `section.tsx` (and the old `config-form`/`train-panel`/`tokenize-panel`) — no dangling imports. Slots are always mounted; empty telemetry shows a muted placeholder, so nothing pops. Verified: production build renders `/training` static; the running dev server hot-reloaded and serves it 200.
