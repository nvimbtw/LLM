"use client"

import { Info } from "lucide-react"

import { Panel } from "@/components/ui/panel"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  type GrammarMetrics,
  type ParadigmResult,
  NO_VALUE,
  formatPercent,
  humanizeParadigm,
} from "@/lib/benchmark"
import { paradigmGloss } from "@/lib/grammar-glossary"

// Per-paradigm accuracy, drawn as a deviation from chance rather than as a bar
// from zero.
//
// 50% is the meaningful midpoint here — it's what a model that has learned nothing
// scores, since every item is a forced choice between two sentences. A bar growing
// from 0 would make a coin flip look like half a result. Growing from the centre
// line makes "learned nothing" read as nothing, which is the truth, and gives
// below-chance paradigms somewhere to go.
export function GrammarPanel({ grammar }: { grammar: GrammarMetrics }) {
  const sorted = [...grammar.paradigms].sort(
    (a, b) => (b.accuracy ?? 0) - (a.accuracy ?? 0)
  )

  return (
    <Panel
      title="Grammar"
      description={`Minimal pairs · ${formatPercent(grammar.accuracy)} overall · ${grammar.pairs} pairs`}
    >
      {sorted.length === 0 ? (
        <p className="py-6 text-center text-sm text-muted-foreground">
          No paradigms evaluated.
        </p>
      ) : (
        // One Provider for the whole list so moving between rows re-opens without
        // re-serving the open delay — the rows read as one glossary, not ten
        // independent popups.
        <TooltipProvider delay={150} closeDelay={80}>
          <ul className="flex flex-col gap-4">
            {sorted.map((paradigm) => (
              <li key={paradigm.name}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className="flex min-w-0 items-baseline gap-1.5">
                    <span className="truncate text-sm font-medium">
                      {humanizeParadigm(paradigm.name)}
                    </span>
                    <ParadigmInfo paradigm={paradigm} />
                  </span>
                  <span className="shrink-0 text-xs text-muted-foreground tabular-nums">
                    {formatPercent(paradigm.accuracy)}
                    <span className="ml-2 text-muted-foreground">
                      {paradigm.pairs}
                    </span>
                  </span>
                </div>
                <DivergingBar accuracy={paradigm.accuracy} />
              </li>
            ))}
          </ul>
        </TooltipProvider>
      )}

      <p className="mt-6 border-t border-border pt-4 text-xs leading-relaxed text-muted-foreground">
        Each pair is two sentences differing in one place, one grammatical. The
        model is correct when it assigns the grammatical one a higher total
        log-probability. Chance is 50%; the centre line marks it.
      </p>
    </Panel>
  )
}

// The hoverable gloss for one paradigm row. A real <button> rather than a bare icon
// so the explanation is reachable by keyboard and by tap, not only by mouse — the
// content here is the only place the paradigm's name is explained, so it can't be
// hover-only.
function ParadigmInfo({ paradigm }: { paradigm: ParadigmResult }) {
  const gloss = paradigmGloss(paradigm.name)

  return (
    <Tooltip>
      {/* TooltipTrigger already renders a <button>, so it is focusable as-is. */}
      <TooltipTrigger
        aria-label={`What ${humanizeParadigm(paradigm.name)} means`}
        className="shrink-0 rounded-full text-muted-foreground/35 transition-colors hover:text-muted-foreground focus-visible:text-muted-foreground focus-visible:ring-1 focus-visible:ring-ring focus-visible:outline-none"
      >
        <Info className="size-3" />
      </TooltipTrigger>
      <TooltipContent
        side="top"
        align="start"
        className="max-w-80 flex-col items-start gap-2 px-3.5 py-3 text-left"
      >
        <p className="text-xs leading-relaxed">
          {gloss?.summary ?? paradigm.description}
        </p>
        {gloss && (
          // The pair itself, marked by a leading ✓ / ✗ rather than by colour alone:
          // the tooltip surface is the inverted foreground, where the theme's
          // good/bad hues don't hold their contrast.
          <div className="flex flex-col gap-1 border-t border-background/20 pt-2 text-xs leading-snug">
            <span className="flex gap-2">
              <span aria-hidden className="opacity-60">
                ✓
              </span>
              <span>{gloss.good}</span>
            </span>
            <span className="flex gap-2 opacity-60">
              <span aria-hidden>✗</span>
              <span className="line-through decoration-background/40">
                {gloss.bad}
              </span>
            </span>
          </div>
        )}
      </TooltipContent>
    </Tooltip>
  )
}

function DivergingBar({ accuracy }: { accuracy: number | null }) {
  if (accuracy === null || !Number.isFinite(accuracy)) {
    return (
      <div
        className="mt-2 h-1.5 w-full rounded-full bg-secondary"
        aria-label={NO_VALUE}
      />
    )
  }

  // Half-width per side, so a paradigm at 100% fills the right half and one at 0%
  // fills the left. `deviation` is the distance from chance in [0, 1].
  const deviation = Math.min(Math.abs(accuracy - 0.5) * 2, 1)
  const above = accuracy >= 0.5
  const width = `${deviation * 50}%`

  return (
    <div className="relative mt-2 h-1.5 w-full rounded-full bg-secondary">
      <div
        className={`absolute top-0 h-full ${above ? "bg-primary" : "bg-destructive"}`}
        style={
          above
            ? {
                left: "50%",
                width,
                borderTopRightRadius: 9999,
                borderBottomRightRadius: 9999,
              }
            : {
                right: "50%",
                width,
                borderTopLeftRadius: 9999,
                borderBottomLeftRadius: 9999,
              }
        }
      />
      {/* The chance line: a hairline tick, not a full rule, so it orients without
          competing with the data. */}
      <div className="absolute top-[-2px] left-1/2 h-[calc(100%+4px)] w-px -translate-x-1/2 bg-border" />
    </div>
  )
}
