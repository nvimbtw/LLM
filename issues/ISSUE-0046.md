---
id: ISSUE-0046
title: "Telemetry restyle — hairline stat run + charts-as-figures with a shared editorial chart theme"
scope: frontend
priority: P2
status: verification
rule: none
location: frontend/components/training/train-panel.tsx frontend/components/training/tokenize-panel.tsx frontend/components/training/chart-theme.ts
blocked-by: [ISSUE-0044]
blocks: []
---

## Description

Part of the `/training` redesign epic (ISSUE-0040). Restyles the live telemetry in
the main column so it reads editorially: the key numbers become a horizontal
**stat run** (hairline-separated, no boxes) and each chart becomes a captioned
**figure** drawn in one shared, theme-matched visual language.

### What's there now

Both panels box every number in `Stat` — a `rounded-lg border bg-card p-3` card
(`train-panel.tsx:43-52`, `tokenize-panel.tsx:26-35`) laid out in a
`grid ... sm:grid-cols-4`. Charts are decent but each configures its axes/grid/
stroke ad-hoc, and the loss chart uses a bespoke warm→cool `epochColor` HSL ramp
(`train-panel.tsx:26-30`) that doesn't reference the site's `--chart-*` tokens
(`globals.css:72-76`). The batch chart hardcodes red/blue/green
(`train-panel.tsx:17-21`).

### The stat run

Replace the boxed `Stat` grid with a single horizontal **stat run**: value +
label pairs laid out inline, separated by thin vertical hairline rules
(`divide-x` with the `border` token), values in `font-heading tabular-nums`, labels
muted and small above/below. No card, no fill — it reads like a figure's data line.
This sits in the shell's stat-run slot (0044), pinned at the top of the main column
so the headline numbers (loss, epoch, avg loss, ETA, progress %) are **at a glance
without scrolling**. Fold the progress percentage into the run (or a thin hairline
progress rule beneath it) rather than a separate boxed `Progress` block.

Keep the mode-awareness: pretrain shows files/chars/vocab/merges; SFT shows
conversations/total/trainable tokens (the existing `isSft` branch in
`tokenize-panel.tsx:63`). During training the train stats take over the run.

### Charts as figures + a shared theme

Add `components/training/chart-theme.ts` as the single source of chart styling and
route every chart through it:

- **Consistent axes/grid.** One set of `XAxis`/`YAxis`/`CartesianGrid` defaults
  (no vertical grid, hairline horizontal lines in the `border` token, muted tick
  labels, `tickLine`/`axisLine` off) so the loss, batch-spread, and vocab charts
  look like one family. Today each redeclares these — centralize them.
- **Theme-matched colors.** Draw from the site `--chart-1..5` tokens
  (`globals.css`), not ad-hoc hex/HSL. The multi-epoch loss ramp can stay a
  *sequential* ramp for the "early→late" read, but derive it from the chart tokens /
  a single documented ramp function in `chart-theme.ts` rather than the inline
  `hsl(12→160)`; the batch min/avg/max lines should map to chart tokens. Preserve
  the epoch legend's early→late gradient swatch idea (`train-panel.tsx:164-178`).
- **Figure framing.** Each chart gets a quiet caption (the descriptive line already
  present, e.g. "Loss per epoch (x = chunk within epoch)") styled as a figure
  caption, and lives in a reserved-height slot so it never causes a jump — keep the
  existing dashed "Waiting for the first loss values…" placeholder pattern
  (`train-panel.tsx:226-230`) and extend it to the secondary charts (render the
  figure frame + caption + a muted placeholder before data, not nothing — the epic's
  "never pop" rule).
- **Preserve all the charting logic.** The epoch pivot/rows, `yDomain` fitting,
  downsampling, `connectNulls`, `isAnimationActive={false}` — all stay. This is a
  skin, not a rewrite of the data plumbing.

> Before touching chart colors/axes, read the `dataviz` skill — it covers
> categorical vs sequential palettes, stat tiles/runs, and light/dark consistency,
> which is exactly this issue's surface.

### Definition of done

- Numbers render as a hairline stat run (no boxed `Stat` cards) pinned at the top of
  the main column; mode-aware fields preserved.
- All three charts draw through `chart-theme.ts` with token-based colors and a
  unified axis/grid style; captions read as figure captions; each has a reserved-
  height placeholder before data (no pop/jump).
- Loss curve, batch spread, and vocab growth are all still correct
  (multi-epoch overlay, fitted y-domain, rolling window). `bun run typecheck` +
  `bun run lint` clean.

## Comments
### Auditor · 2026-07-12
Child of ISSUE-0040, blocked by the shell (0044) which owns the stat-run/figure
slots. Parallel with 0045/0047. Scoped to visual language (stat run + shared chart
theme) with the data/plumbing held constant.

### Builder · 2026-07-12
Added `components/training/chart-theme.ts` (shared `axisProps`/`yAxisProps`/`gridProps`, single-hue sequential ramps) + `stat-run.tsx` (`StatRun`, `Figure`, `FigurePlaceholder`) + `telemetry.tsx`. Boxed `Stat` cards replaced by a hairline stat run pinned at the top of the main column (mode/phase-aware: tokenize→train). All three charts route through the theme with reserved-height captioned figures + placeholders. Per the dataviz skill: the loss "early→late" overlay and the min/avg/max spread are *magnitude*, so both now use a single-hue light→dark sequential ramp (killed the old warm→cool rainbow and the hardcoded red/blue/green); vocab keeps the `--chart-2` token. Loss pivot/yDomain/connectNulls logic preserved.
