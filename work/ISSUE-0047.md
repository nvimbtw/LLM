---
id: ISSUE-0047
type: issue
title: "Appendix drawer — log + checkpoints in a persistent collapsible detail region"
scope: frontend
priority: P3
status: verification
rule: none
location: frontend/components/training/appendix.tsx frontend/components/training/tokenize-panel.tsx frontend/components/training/train-panel.tsx
blocked-by: [ISSUE-0044]
blocks: []
---

## Description

Part of the `/training` redesign epic (EPIC-0040). Tucks the low-frequency detail —
the tokenizer **log** and the **checkpoint** list — into a single persistent
**appendix** at the foot of the main column, so the important telemetry stays
uncluttered while the details remain one click away.

### What's there now

The log lives inside `tokenize-panel.tsx` behind a show/hide toggle
(`tokenize-panel.tsx:118-141`); the checkpoint list lives inside `train-panel.tsx`
(`train-panel.tsx:286-297`). They're scattered across two panels and the log's
reveal grows the panel inline.

### The appendix

A new `components/training/appendix.tsx` — one collapsible region living in the
shell's appendix slot (0044), holding both:

- **Log** — the raw tokenizer/build output (currently `log: string[]`), monospace,
  scrollable within a fixed max-height, auto-scrolled to the newest line (preserve
  the `logEndRef` scroll-into-view behaviour from `tokenize-panel.tsx:47-51`). Show
  the line count in the header.
- **Checkpoints** — the saved checkpoint paths (`checkpoints: string[]`), monospace,
  as today.

Presentation, per the editorial theme:

- A single hairline-headed disclosure (`▸ Appendix — log · checkpoints`), collapsed
  by default. It's the *one* place progressive-disclosure detail is allowed, framed
  as an article appendix.
- **Persistent slot, never pop.** The appendix header is always present in its slot
  (even with an empty log / no checkpoints — show a muted "No output yet");
  expanding/collapsing it must not reflow the telemetry above it — it opens
  *downward* at the page foot, its body scrolls internally with a fixed max-height,
  so neighbours don't move. This is the "reserve the slot, fill it in" rule.
- Reuse a shadcn disclosure primitive if one exists (`components/ui/`); otherwise a
  simple controlled toggle like the current log button is fine — just don't
  hand-roll a `Card`.

### Rules

- Move the log + checkpoint rendering *out* of `tokenize-panel.tsx` /
  `train-panel.tsx` and into `appendix.tsx`; the panels shrink to pure telemetry.
  The `log`/`checkpoints` data still comes from the `useTraining` hook — pass them
  into the appendix from the shell, don't duplicate state.
- Keep the log capped/scrolled (the hook already caps at 500 lines —
  `use-training.ts:149`); the appendix just needs a fixed-height scroll container.

### Definition of done

- Log and checkpoints render only inside the appendix; the telemetry panels no
  longer contain them.
- The appendix is always present, collapsed by default, and opening it does not
  move the charts/stats above it.
- Log auto-scrolls to newest when open; checkpoints list correctly.
  `bun run typecheck` + `bun run lint` clean.

## Comments
### Auditor · 2026-07-12
Child of EPIC-0040, blocked by the shell (0044) which owns the appendix slot.
P3 — it's the "tuck the noise away" polish, lower-stakes than the shell/rail/figures.
Parallel with 0045/0046.

### Builder · 2026-07-12
Added `components/training/appendix.tsx` — a single always-present collapsible at the foot of the main column holding the log (monospace, fixed max-h scroll, auto-scroll to newest) and checkpoints, both moved out of the telemetry panels. Opens downward with internal scroll so expanding never reflows the charts above. Empty states show muted captions.
