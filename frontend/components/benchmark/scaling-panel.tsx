"use client"

import { Panel } from "@/components/ui/panel"
import { type BenchmarkRun, formatCompact } from "@/lib/benchmark"

import { type ScorePoint, ScoreScatter, logDomain, logTicks } from "./score-scatter"

// Score against training tokens. The question it answers is the one the whole
// benchmark exists for: is more data buying anything, and is it still buying it at the
// current corpus size? See `score-scatter.tsx` for the figure's construction.
export function ScalingPanel({ runs }: { runs: BenchmarkRun[] }) {
  // A run needs both coordinates to be plottable. A missing token count means the
  // model's config predates the field — dropping the run is honest; plotting it at
  // zero would invent the strongest possible claim about a tiny corpus.
  const points: ScorePoint[] = runs
    .filter((run) => run.status === "done" && run.score !== null && run.train_tokens !== null)
    .map((run) => ({
      x: run.train_tokens as number,
      score: run.score as number,
      name: run.model_name,
      created_at: run.created_at,
    }))
    // Plot order doesn't affect a scatter's reading, but keeping it stable keeps
    // overlapping marks stacked the same way between renders.
    .sort((a, b) => a.x - b.x)

  if (points.length < 2) {
    return (
      <Panel title="Scaling" description="Score vs training tokens">
        <p className="py-6 text-center text-sm text-muted-foreground">
          {points.length === 0
            ? "No scored runs with a known corpus size yet."
            : "One point so far — benchmark a model trained on a different amount of data to have something to compare it against."}
        </p>
      </Panel>
    )
  }

  const domain = logDomain(points.map((p) => p.x))

  return (
    <Panel title="Scaling" description={`Score vs training tokens · ${points.length} models`}>
      <ScoreScatter
        points={points}
        domain={domain}
        ticks={logTicks(domain)}
        formatX={formatCompact}
        xLabel="training tokens"
      />

      <p className="mt-4 border-t border-border/60 pt-4 text-[11px] leading-relaxed text-muted-foreground/70">
        Tokens multiply to the left and divide to the right — each labelled step is a
        fixed factor, not a fixed amount. Every point is a separate model, not a stage
        of one, so they&apos;re deliberately left unjoined: nothing travelled between
        them. A point sitting above its neighbours got more out of its corpus than the
        corpus size alone predicts.
      </p>
    </Panel>
  )
}
