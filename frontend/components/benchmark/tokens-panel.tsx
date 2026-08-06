"use client"

import { Panel } from "@/components/ui/panel"
import {
  type BenchmarkRun,
  compactAxisFormat,
  MIN_CONVERSATIONS,
} from "@/lib/benchmark"
import { TIER_LABEL } from "@/lib/jobs"

import {
  type ScorePoint,
  ScoreScatter,
  logDomain,
  logTicks,
} from "./score-scatter"

// Score against the tokens the run spent to produce it — the same figure as the
// duration panel with a different cost on x, and the question it asks is what a
// point of score costs in work rather than in wall clock. See `score-scatter.tsx`
// for the figure's construction.
//
// This used to plot training tokens, which described the *model* and not the run:
// it never moved between two benchmarks of the same model however differently they
// were configured, and it said nothing about what the measurement cost.
// `eval_tokens` is every token the model actually forwarded — corpus scoring, each
// minimal pair, each sampled reply — counted at the inference session, so it is the
// same number on any machine, which is the one thing the duration cannot claim.
export function TokensPanel({
  runs,
  filtered,
}: {
  /** The runs to plot — already narrowed to the models the filter has ticked. */
  runs: BenchmarkRun[]
  /** Some models are switched off, so an empty plot is a filter, not a missing history. */
  filtered?: boolean
}) {
  // A run needs both coordinates to be plottable. A missing token count means the
  // run predates the counter — dropping it is honest; plotting it at zero would
  // invent the strongest possible claim about a free benchmark.
  const points: ScorePoint[] = runs
    .filter(
      (run) =>
        run.status === "done" && run.score !== null && run.eval_tokens !== null
    )
    .map((run) => ({
      x: run.eval_tokens as number,
      score: run.score as number,
      // A swept run's dot is labelled by its budget, not by its model: the three
      // points of one sweep are the same model, and three identical names beside
      // three joined dots would say nothing the line has not already said. The model
      // rides along in `model` and is drawn once per line, on its anchor point.
      name: run.think_budget ? TIER_LABEL[run.think_budget] : run.model_name,
      model: run.model_name,
      created_at: run.created_at,
      series: run.id_series,
      budget: run.think_budget,
      conversations: run.eval_conversations,
      // Thin only when there is a count to be thin: a language run scores no
      // conversations at all, and `null` there means "not applicable", not "zero".
      provisional:
        run.eval_conversations !== null &&
        run.eval_conversations < MIN_CONVERSATIONS,
    }))
    // Plot order doesn't affect a scatter's reading, but keeping it stable keeps
    // overlapping marks stacked the same way between renders.
    .sort((a, b) => a.x - b.x)

  if (points.length < 2) {
    return (
      <Panel
        title="Tokens"
        description="Score vs tokens used to complete the run"
      >
        <p className="py-6 text-center text-sm text-muted-foreground">
          {filtered
            ? "Not enough shown to compare — tick another model above."
            : points.length === 0
              ? "No completed runs with a token count yet — runs from before this was measured aren't plotted."
              : "One point so far — benchmark another model to have something to compare it against."}
        </p>
      </Panel>
    )
  }

  const domain = logDomain(points.map((p) => p.x))
  const ticks = logTicks(domain)
  // Labels precise enough for the range on screen: these runs land within a fraction
  // of a percent of each other, and one-decimal compact labels would print the same
  // number five times across the axis.
  const formatX = compactAxisFormat(ticks)

  return (
    <Panel
      title="Tokens"
      description={`Score vs tokens used to complete the run · ${points.length} runs`}
    >
      <ScoreScatter
        points={points}
        domain={domain}
        ticks={ticks}
        formatX={formatX}
        xLabel="tokens used"
      />

      <p className="mt-4 border-t border-border pt-4 text-xs leading-relaxed text-muted-foreground">
        Tokens multiply to the left and divide to the right — each labelled step
        is a fixed factor, not a fixed amount. What a run spends is set by the
        model as much as by the suite: a longer context window scores the same
        held-out text in fewer, longer windows, and a model that generates
        freely pays for every token it writes. Points are deliberately left
        unjoined — each is a separate run, not a stage of one. A point sitting
        above its neighbours bought its score more cheaply. The axis fits the
        runs shown, so read the labels before the spread: fine-tunes of one
        model differ by a fraction of a percent here and still fill the plot. A{" "}
        <strong>
          joined line is one model swept across the three thinking budgets
        </strong>{" "}
        — the one case where connecting dots is honest, since a single model
        really did travel that path as it was told to think harder. Each line is
        named once, at its long-budget end; its dots carry the budgets. A hollow
        dot rests on fewer than {MIN_CONVERSATIONS} held-out conversations;
        hover it for the count.
      </p>
    </Panel>
  )
}
