"use client"

import * as React from "react"
import { Check } from "lucide-react"
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"

import { Progress } from "@/components/ui/progress"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"
import {
  axisProps,
  BUILD_COLOR,
  gridProps,
  yAxisProps,
} from "@/components/training/chart-theme"
import {
  compact,
  duration,
  Figure,
  FigurePlaceholder,
} from "@/components/training/stat-run"
import { cn } from "@/lib/utils"
import type { BuildPoint } from "@/hooks/use-training"
import type { BuildSource, BuildStep, TokenizeStats } from "@/lib/training"

// The steps a pretrain's phase 1 is made of, in the order the engine enters
// them (`llm.train.progress.BUILD_SPANS`). A fine-tune's build is one step with
// three counters rather than three steps, so it gets its own list below.
const PRETRAIN_STEPS: BuildStep[] = [
  "sampling",
  "merges",
  "packing",
  "measuring",
]

// How often the elapsed clock advances between frames. A second, because that is
// the resolution the reading is shown at — anything finer is a re-render nobody
// can see, and anything coarser is a clock that visibly stutters.
const CLOCK_MS = 1000

const rateConfig = {
  rate: { label: "Tokens / sec", color: BUILD_COLOR },
} satisfies ChartConfig

export interface BuildProgressProps {
  isSft: boolean
  tokenize: Partial<TokenizeStats>
  series: BuildPoint[]
}

/**
 * What phase 1 is doing right now, while it does it.
 *
 * This slot used to hold a three-line checklist whose states were *inferred*
 * from which counts had arrived — which worked for the two steps that reported
 * one, and read the longest step of the run as an already-finished tokenizer.
 * The engine now names the step it is on and says how far into it it is, so
 * everything here is read rather than guessed.
 *
 * Three things, in the order the question is usually asked: which step, how far
 * through, and how fast — then which source that leaves it on. Every one of them
 * renders its own absence, because a build reports what it has measured and
 * nothing else, and a dash is the difference between a figure nobody took and a
 * figure of zero.
 */
export function BuildProgress({ isSft, tokenize, series }: BuildProgressProps) {
  const steps = isSft ? sftSteps(tokenize) : pretrainSteps(tokenize)
  const elapsed = tokenize.elapsed_secs

  return (
    <div className="space-y-6">
      <Figure
        caption={
          isSft
            ? "Dataset build — rendering chat conversations into the masked SFT dataset"
            : "Corpus build — a vocabulary, then the corpus packed under it"
        }
        aside={
          <span className="tabular-nums">
            {elapsed != null ? (
              <BuildClock
                key={elapsed}
                from={elapsed}
                done={tokenize.done === true}
              />
            ) : (
              "—"
            )}{" "}
            elapsed
            {tokenize.eta_secs != null && tokenize.done !== true && (
              <> · {duration(tokenize.eta_secs)} left</>
            )}
          </span>
        }
        minHeight="0"
      >
        <ul className="rounded-(--radius-control-sm) border border-dashed px-4 py-1">
          {steps.map((step) => (
            <BuildStepRow key={step.label} {...step} />
          ))}
        </ul>
      </Figure>

      <Figure
        caption="Tokens a second, over the build's own clock — the line a stalled read flattens before any counter stops moving"
        aside={
          tokenize.tokens_per_sec != null ? (
            <span className="tabular-nums">
              {compact(tokenize.tokens_per_sec)} / s now
            </span>
          ) : undefined
        }
        minHeight="12rem"
      >
        {series.length > 1 ? (
          <ChartContainer
            config={rateConfig}
            className="aspect-auto h-48 w-full"
          >
            <LineChart data={series} margin={{ left: 4, right: 4, top: 4 }}>
              <CartesianGrid {...gridProps} />
              <XAxis
                dataKey="elapsed"
                type="number"
                domain={["dataMin", "dataMax"]}
                tickFormatter={(v: number) => duration(v)}
                {...axisProps}
              />
              <YAxis
                tickFormatter={(v: number) => compact(v)}
                {...yAxisProps}
              />
              <ChartTooltip
                cursor={false}
                content={
                  <ChartTooltipContent
                    labelFormatter={(v) => `${duration(Number(v))} in`}
                    formatter={(v) => `${compact(Number(v))} tokens / sec`}
                  />
                }
              />
              <Line
                dataKey="rate"
                name="Tokens / sec"
                type="linear"
                stroke={BUILD_COLOR}
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ChartContainer>
        ) : (
          <FigurePlaceholder className="h-48">
            {tokenize.step === "merges"
              ? "Learning the merges — one call into the tokenizer library, which reports nothing until it returns"
              : "Waiting for the first throughput reading…"}
          </FigurePlaceholder>
        )}
      </Figure>

      <SourceStrip
        sources={tokenize.sources}
        total={tokenize.sources_total}
      />
    </div>
  )
}

// One step of the build: a mark, what it is, and what it has produced so far.
function BuildStepRow({
  label,
  value,
  state,
  fraction,
}: {
  label: string
  value: string
  state: "done" | "active" | "todo"
  fraction?: number
}) {
  return (
    <li className="border-b border-border/40 py-2.5 last:border-b-0">
      <div className="flex items-baseline justify-between gap-4">
        <span className="flex min-w-0 items-center gap-2">
          <span
            aria-hidden
            className={cn(
              "flex size-4 shrink-0 items-center justify-center rounded-full",
              state === "done" && "bg-chart-2/20 text-chart-2",
              state === "active" &&
                "animate-pulse bg-primary/30 ring-1 ring-primary/50",
              state === "todo" && "bg-muted"
            )}
          >
            {state === "done" && <Check className="size-2.5" />}
          </span>
          <span
            className={cn(
              "truncate text-sm",
              state === "todo" ? "text-muted-foreground" : "text-foreground"
            )}
          >
            {label}
          </span>
        </span>
        <span className="shrink-0 text-sm text-muted-foreground tabular-nums">
          {value}
        </span>
      </div>
      {/* Only the step in flight draws one. A bar on a finished step is a full
          bar, which says nothing, and a bar on one that has not started is an
          empty one, which says something untrue. */}
      {state === "active" && fraction != null && (
        <Progress className="mt-2" value={Math.round(fraction * 100)} />
      )}
    </li>
  )
}

// Where the build has got to, source by source, in reading order.
function SourceStrip({
  sources,
  total,
}: {
  sources: BuildSource[] | undefined
  total: number | undefined
}) {
  if (!sources?.length) return null
  const shown = sources.length
  const named = total != null && total > shown
  // The stage only earns a label on a run that has more than one. A plain run is
  // stage 1 on every row, which is a column of the same word.
  const staged = sources.some((source) => source.stage > 0)
  return (
    <Figure
      caption="Documents read per source, in the order the pack reads them — with what each is costing in characters per token"
      aside={
        named ? (
          <span className="tabular-nums">
            {shown} of {total} — the window follows the source being read
          </span>
        ) : undefined
      }
      minHeight="0"
    >
      <ul className="space-y-2.5">
        {sources.map((source) => (
          <li key={`${source.stage}-${source.dataset}`}>
            <div className="flex items-baseline justify-between gap-3 text-xs">
              <span className="min-w-0 truncate">
                <span className="text-foreground">{source.dataset}</span>
                {staged && (
                  <span className="ml-1.5 text-muted-foreground">
                    stage {source.stage + 1}
                  </span>
                )}
              </span>
              <span className="shrink-0 text-muted-foreground tabular-nums">
                {compact(source.documents)}
                {source.expected != null && ` / ${compact(source.expected)}`}
                {source.tokens > 0 &&
                  ` · ${(source.chars / source.tokens).toFixed(2)} c/t`}
              </span>
            </div>
            <Progress
              className="mt-1"
              value={
                source.expected
                  ? Math.min(
                      100,
                      Math.round((source.documents / source.expected) * 100)
                    )
                  : 0
              }
            />
          </li>
        ))}
      </ul>
    </Figure>
  )
}

/**
 * The build's elapsed clock, carried on between the frames that set it.
 *
 * Remounted by its `key` whenever a frame lands, so it counts up from the
 * engine's own reading rather than from when this page opened: a page attached
 * halfway through a build reads the build's age, and the two can never drift
 * apart however long the gap between frames grows.
 *
 * The gaps are the point. Phase 1's frame interval widens as it runs, and the
 * step that learns the merges cannot report at all — it is one call into a
 * library with no callback — so for minutes at a time this is the only thing on
 * the page still moving. A finished build stops where it stopped: a clock still
 * running is a claim that something still is.
 */
function BuildClock({ from, done }: { from: number; done: boolean }) {
  const [ticks, setTicks] = React.useState(0)

  React.useEffect(() => {
    if (done) return
    const id = window.setInterval(() => setTicks((t) => t + 1), CLOCK_MS)
    return () => window.clearInterval(id)
  }, [done])

  return <>{duration(from + (done ? 0 : ticks * (CLOCK_MS / 1000)))}</>
}

// A step is done once the build has moved past it, active while it is on it, and
// to come otherwise. `reusing` is the build finding its corpus already on disk:
// everything that would have produced it is done, and nothing was run.
function stateOf(
  step: BuildStep,
  current: BuildStep | undefined,
  done: boolean
): "done" | "active" | "todo" {
  if (done) return "done"
  if (current == null) return "todo"
  if (current === "reusing") return step === "measuring" ? "active" : "done"
  const at = PRETRAIN_STEPS.indexOf(current)
  const mine = PRETRAIN_STEPS.indexOf(step)
  if (mine < at) return "done"
  return mine === at ? "active" : "todo"
}

function pretrainSteps(tokenize: Partial<TokenizeStats>) {
  const done = tokenize.done === true
  const state = (step: BuildStep) => stateOf(step, tokenize.step, done)
  const drawn = tokenize.sampled_bytes
  const budget = tokenize.sample_budget
  const read = tokenize.documents_done
  const reused = tokenize.step === "reusing"

  return [
    {
      label: "Vocabulary sample drawn",
      value:
        drawn == null
          ? "—"
          : budget
            ? `${megabytes(drawn)} of ${megabytes(budget)}`
            : megabytes(drawn),
      state: state("sampling"),
      fraction: drawn != null && budget ? drawn / budget : undefined,
    },
    {
      label: "Merges learned",
      value:
        tokenize.vocab != null
          ? `${compact(tokenize.vocab)} vocab · ${compact(tokenize.merges)} merges`
          : "—",
      state: state("merges"),
    },
    {
      label: "Corpus packed",
      value: reused
        ? "reused — packed by an earlier run"
        : read != null && tokenize.documents != null
          ? `${compact(read)} / ${compact(tokenize.documents)} docs · ${compact(tokenize.total_tokens)} tokens`
          : tokenize.total_tokens != null
            ? `${compact(tokenize.total_tokens)} tokens`
            : "—",
      state: state("packing"),
      fraction:
        read != null && tokenize.documents
          ? read / tokenize.documents
          : undefined,
    },
    {
      label: "Corpus measured",
      value:
        tokenize.boundary_straddle != null
          ? `${compact(tokenize.documents)} documents · ${(tokenize.boundary_straddle * 100).toFixed(0)}% of windows cross one`
          : "—",
      state: state("measuring"),
    },
  ]
}

// A fine-tune's build is one step producing three counters, not three steps — so
// the rows below the first are readings of the step in flight rather than stages
// of it, and none of them is finished until the build is.
function sftSteps(tokenize: Partial<TokenizeStats>) {
  const read = tokenize.documents_done
  const rendered = tokenize.conversations ?? tokenize.examples
  const done = tokenize.done === true
  const reused = tokenize.step === "reusing"
  const state = (has: boolean): "done" | "active" | "todo" =>
    done ? "done" : has ? "active" : "todo"

  return [
    {
      label: "Conversations rendered",
      value: reused
        ? "reused — rendered by an earlier run"
        : rendered == null
          ? read != null
            ? `${compact(read)} read`
            : "—"
          : tokenize.documents != null
            ? `${compact(rendered)} of ${compact(tokenize.documents)} read`
            : compact(rendered),
      state: state(true),
      fraction:
        read != null && tokenize.documents
          ? read / tokenize.documents
          : undefined,
    },
    {
      label: "Tokens written",
      value: compact(tokenize.total_tokens),
      state: state(tokenize.total_tokens != null),
    },
    {
      label: "Trainable (assistant replies)",
      value: compact(tokenize.trainable_tokens),
      state: state(tokenize.trainable_tokens != null),
    },
  ]
}

// The vocabulary's sample is the one figure here measured in text rather than in
// documents or tokens, and it is bounded in the megabytes `bpe.SAMPLE_BYTES`
// names — so it reads as megabytes and not as a compacted count of bytes.
function megabytes(bytes: number): string {
  return `${(bytes / 1e6).toFixed(1)} MB`
}
