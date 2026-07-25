"use client"

import { Panel } from "@/components/ui/panel"
import { type BenchmarkRun, formatDurationCompact } from "@/lib/benchmark"

import { type ScorePoint, ScoreScatter, logDomain, pickDensest } from "./score-scatter"

// Round durations, in seconds, sparsest tier first. Time isn't round at powers of ten —
// the generic 1/2/5 ladder labels a 40-minute axis "33m 20s · 16m 40s · 8m 20s", which
// are exact but not values anyone thinks in. Every entry below formats to a whole unit
// or a familiar half (30s, 1m 30s, 45m), and `pickDensest` takes the finest tier that
// still fits the axis — coarse tiers for a run range spanning hours, the 15-second
// granularity of the last tier for the minute-ish runs a small model actually takes.
const DURATION_TIERS = [
  [1, 10, 60, 600, 3600, 86_400],
  [1, 5, 10, 30, 60, 300, 600, 1800, 3600, 21_600, 86_400],
  [1, 2, 5, 10, 15, 30, 45, 60, 120, 300, 600, 900, 1800, 3600, 7200, 21_600, 43_200, 86_400],
  [5, 10, 15, 20, 30, 45, 60, 75, 90, 120, 150, 180, 240, 300, 450, 600, 900, 1200, 1800, 2700, 3600, 5400, 7200],
]

function durationTicks(domain: [number, number]): number[] {
  const [min, max] = domain
  return pickDensest(
    DURATION_TIERS.map((tier) => tier.filter((value) => value >= min && value <= max)),
    domain
  )
}

// Score against how long the run took — the same figure as the scaling panel with a
// different cost on x, and the question it asks is what a point of score costs in
// time rather than in data.
//
// The duration is the run's own wall clock: the gap between the row being opened and
// being completed. That tracks the things that make a model expensive to run at all —
// parameter count, context length, vocabulary — so a model far to the left is one that
// took real work to score. Two caveats it can't see: the clock includes any time the
// run spent waiting on the GPU, and it's measured on whatever hardware was to hand, so
// durations are only comparable between runs on the same machine.
export function DurationPanel({ runs }: { runs: BenchmarkRun[] }) {
  // Both timestamps are needed, and only a finished run has the second one. A
  // zero-or-negative gap (a clock adjustment, a run that completed inside the same
  // millisecond) has no place on a log axis, so it's dropped rather than clamped.
  const points: ScorePoint[] = runs
    .filter((run) => run.status === "done" && run.score !== null && run.completed_at !== null)
    .map((run) => ({
      x: (new Date(run.completed_at as string).getTime() - new Date(run.created_at).getTime()) / 1000,
      score: run.score as number,
      name: run.model_name,
      created_at: run.created_at,
    }))
    .filter((point) => point.x > 0)
    // Plot order doesn't affect a scatter's reading, but keeping it stable keeps
    // overlapping marks stacked the same way between renders.
    .sort((a, b) => a.x - b.x)

  if (points.length < 2) {
    return (
      <Panel title="Cost" description="Score vs time to complete">
        <p className="py-6 text-center text-sm text-muted-foreground">
          {points.length === 0
            ? "No completed runs to time yet."
            : "One point so far — benchmark another model to have something to compare it against."}
        </p>
      </Panel>
    )
  }

  const domain = logDomain(points.map((p) => p.x))

  return (
    <Panel title="Cost" description={`Score vs time to complete · ${points.length} models`}>
      <ScoreScatter
        points={points}
        domain={domain}
        ticks={durationTicks(domain)}
        formatX={formatDurationCompact}
        xLabel="to complete"
      />

      <p className="mt-4 border-t border-border/60 pt-4 text-[11px] leading-relaxed text-muted-foreground/70">
        Time multiplies to the left: the slowest runs are the leftmost, and each
        labelled step is a fixed factor. A point high and to the right earned its score
        cheaply. Wall clock, so it includes any wait for the GPU and is only comparable
        between runs on the same machine.
      </p>
    </Panel>
  )
}
