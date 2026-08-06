"use client"

import * as React from "react"
import { Cell, Pie, PieChart } from "recharts"

import { ChartContainer, type ChartConfig } from "@/components/ui/chart"
import { tierColors } from "@/components/training/chart-theme"
import { TIER_LABEL, type Tier } from "@/lib/jobs"
import { cn } from "@/lib/utils"

const TIERS: Tier[] = ["low", "med", "high"]

const chartConfig = {
  answers: { label: "Answers" },
} satisfies ChartConfig

/**
 * How the answers divided across the three budgets.
 *
 * **A part-to-whole reading, which is the one job a donut does well**: three
 * ordered segments, each labelled with its own count and share, summing to a total
 * that sits in the hole. It replaced three bars that answered "which is biggest"
 * — a question nobody asks of this run. What you actually want to see is whether
 * one budget is swallowing the corpus, and a ring shows that in one glance.
 *
 * The colours are a **sequential ramp, not three hues**: short/medium/long is an
 * ordered scale, so more thinking is more ink. Every segment carries its number
 * beside it, which is also what relieves the lightest step's sub-3:1 contrast
 * against the surface — see the ramp's note in `app/globals.css`.
 */
export function TierDonut({ perTier }: { perTier: [string, number][] }) {
  const counts = new Map(perTier)
  const data = TIERS.map((tier) => ({
    tier,
    label: TIER_LABEL[tier],
    value: counts.get(tier) ?? 0,
  }))
  const total = data.reduce((n, d) => n + d.value, 0)

  if (total === 0) {
    return (
      <p className="py-10 text-center text-sm text-muted-foreground">
        Nothing filed yet.
      </p>
    )
  }

  // An even split is the run working; the distance from it is the story. Reported
  // as the largest share so the sentence below reads as a verdict rather than as
  // three more numbers the reader has to compare themselves.
  const leader = data.reduce((a, b) => (b.value > a.value ? b : a))
  const leaderShare = Math.round((leader.value / total) * 100)

  return (
    <div className="flex flex-col items-center gap-5 sm:flex-row sm:items-center sm:gap-6">
      <div className="relative shrink-0">
        <ChartContainer
          config={chartConfig}
          className="aspect-square h-[168px] w-[168px]"
        >
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="label"
              innerRadius={52}
              outerRadius={80}
              // A real gap in the surface rather than a stroke around each arc:
              // the separation is negative space, not extra ink.
              paddingAngle={2}
              strokeWidth={0}
              isAnimationActive={false}
            >
              {data.map((d) => (
                <Cell key={d.tier} fill={tierColors[d.tier]} />
              ))}
            </Pie>
          </PieChart>
        </ChartContainer>

        {/* The total lives in the hole — the figure's hero number, and the thing
            the three shares are shares *of*. Pointer-events off so it never
            intercepts a hover meant for the ring. */}
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-display text-2xl font-semibold tabular-nums">
            {total.toLocaleString()}
          </span>
          <span className="text-xs text-muted-foreground">answers</span>
        </div>
      </div>

      {/* The legend doubles as the table view: swatch, name, count, share. Every
          number the ring encodes is also written down, so the figure never has to
          be read by colour alone. */}
      <dl className="min-w-0 flex-1 space-y-2.5">
        {data.map((d) => (
          <div key={d.tier} className="flex items-baseline gap-2.5">
            <span
              aria-hidden
              className="size-2.5 shrink-0 rounded-full"
              style={{ background: tierColors[d.tier] }}
            />
            <dt className="min-w-0 flex-1 truncate text-sm">{d.label}</dt>
            <dd className="text-sm tabular-nums">
              {d.value.toLocaleString()}
              <span className="ml-1.5 text-xs text-muted-foreground">
                {Math.round((d.value / total) * 100)}%
              </span>
            </dd>
          </div>
        ))}
        <p className="border-t border-border pt-2.5 text-xs leading-relaxed text-muted-foreground">
          {leaderShare >= 60 ? (
            <>
              <strong className="font-medium text-foreground">
                {leader.label.toLowerCase()} holds {leaderShare}%
              </strong>{" "}
              of the corpus. One budget this dominant means the teacher is
              writing to a length of its own choosing, and the marker is
              learning to mean less than it should.
            </>
          ) : (
            <>
              No budget holds more than {leaderShare}% — the three separated,
              which is the run working. What the counts cannot say is whether
              the short answers are as <em>good</em> as the long ones; only the
              replies below can.
            </>
          )}
        </p>
      </dl>
    </div>
  )
}

/**
 * How many of the three budgets each question actually got answered at.
 *
 * **The figure that says whether this run produced triples at all.** A question is
 * accepted as soon as one of its three answers survives, so the headline count is
 * questions, not triples — and on the corpus that prompted this panel, 211
 * questions carried just 5 that covered all three budgets. That gap is invisible in
 * every other number on the page and it is the one that decides whether the data
 * teaches the student anything about length.
 *
 * Emphasis rather than a ramp: the all-three bar is the story, the other two are
 * context, so one bar carries the accent and the rest recede. The tier ramp would
 * be wrong here — its colours mean *budget*, and nothing on this figure is a budget.
 */
export function CoverageBars({
  byTiers,
  className,
}: {
  byTiers: number[]
  className?: string
}) {
  const rows = [
    {
      label: "One budget",
      value: byTiers[0] ?? 0,
      hint: "an ordinary conversation carrying a marker",
    },
    {
      label: "Two budgets",
      value: byTiers[1] ?? 0,
      hint: "a pair — a contrast, but not the full one",
    },
    {
      label: "All three",
      value: byTiers[2] ?? 0,
      hint: "a real triple: one question, three lengths",
    },
  ]
  const total = rows.reduce((n, r) => n + r.value, 0)
  if (total === 0) {
    return (
      <p
        className={cn(
          "py-10 text-center text-sm text-muted-foreground",
          className
        )}
      >
        No questions accepted yet.
      </p>
    )
  }
  const max = Math.max(...rows.map((r) => r.value), 1)
  const complete = rows[2].value
  const completeShare = Math.round((complete / total) * 100)

  return (
    <div className={cn("space-y-4", className)}>
      <div className="space-y-3">
        {rows.map((row, i) => {
          const emphasised = i === 2
          return (
            <div key={row.label} className="space-y-1">
              <div className="flex items-baseline justify-between gap-3">
                <span
                  className={cn(
                    "text-sm",
                    !emphasised && "text-muted-foreground"
                  )}
                >
                  {row.label}
                </span>
                <span className="text-sm tabular-nums">
                  {row.value.toLocaleString()}
                  <span className="ml-1.5 text-xs text-muted-foreground">
                    {Math.round((row.value / total) * 100)}%
                  </span>
                </span>
              </div>
              {/* Thin marks, rounded data-end, grown from one baseline. The track
                  is the same width for every row so the bars are comparable by
                  length rather than by their own scale. */}
              <div className="h-2 overflow-hidden rounded-full bg-muted">
                <div
                  className={cn(
                    "h-full rounded-full",
                    emphasised ? "bg-kind-teach" : "bg-muted-foreground/35"
                  )}
                  style={{ width: `${(row.value / max) * 100}%` }}
                />
              </div>
              <p className="text-xs text-muted-foreground">{row.hint}</p>
            </div>
          )
        })}
      </div>

      <p className="border-t border-border pt-3 text-xs leading-relaxed text-muted-foreground">
        {completeShare >= 50 ? (
          <>
            <strong className="font-medium text-foreground">
              {completeShare}% of questions are complete triples.
            </strong>{" "}
            Most of what this run wrote can teach the student that a budget
            changes the answer rather than the question.
          </>
        ) : (
          <>
            <strong className="font-medium text-foreground">
              Only {completeShare}% of questions are complete triples.
            </strong>{" "}
            The rest lost an answer to validation — a malformed reply, a
            misquoted number, or a conversation too long for the student&apos;s
            window. Each survivor is still usable SFT data; it just is not a
            contrast.
          </>
        )}
      </p>
    </div>
  )
}

/**
 * Why answers were thrown away, worst first.
 *
 * Horizontal because the labels are sentences — the reason is the data here, and a
 * column chart would turn every one of them into rotated text. This is the panel's
 * diagnostic: a run whose output is nearly all discarded for one reason is a
 * different problem from one discarding a little of everything, and a single
 * "Discarded: 412" cannot tell the two apart.
 */
export function RejectionBars({ rejected }: { rejected: [string, number][] }) {
  if (rejected.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        Nothing was discarded — every answer the teacher wrote passed
        validation.
      </p>
    )
  }

  const sorted = [...rejected].sort((a, b) => b[1] - a[1])
  const total = sorted.reduce((n, [, count]) => n + count, 0)
  const max = Math.max(...sorted.map(([, count]) => count), 1)

  return (
    <div className="space-y-3">
      {sorted.map(([reason, count]) => (
        <div key={reason} className="space-y-1">
          <div className="flex items-baseline justify-between gap-3">
            <span className="min-w-0 flex-1 text-sm">{reason}</span>
            <span className="shrink-0 text-sm tabular-nums">
              {count.toLocaleString()}
              <span className="ml-1.5 text-xs text-muted-foreground">
                {Math.round((count / total) * 100)}%
              </span>
            </span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-muted-foreground/45"
              style={{ width: `${(count / max) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}
