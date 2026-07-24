import { Panel } from "@/components/ui/panel"
import {
  type GrammarMetrics,
  NO_VALUE,
  formatPercent,
  humanizeParadigm,
} from "@/lib/benchmark"

// Per-paradigm accuracy, drawn as a deviation from chance rather than as a bar
// from zero.
//
// 50% is the meaningful midpoint here — it's what a model that has learned nothing
// scores, since every item is a forced choice between two sentences. A bar growing
// from 0 would make a coin flip look like half a result. Growing from the centre
// line makes "learned nothing" read as nothing, which is the truth, and gives
// below-chance paradigms somewhere to go.
export function GrammarPanel({ grammar }: { grammar: GrammarMetrics }) {
  const sorted = [...grammar.paradigms].sort((a, b) => (b.accuracy ?? 0) - (a.accuracy ?? 0))

  return (
    <Panel
      title="Grammar"
      description={`Minimal pairs · ${formatPercent(grammar.accuracy)} overall · ${grammar.pairs} pairs`}
    >
      {sorted.length === 0 ? (
        <p className="py-6 text-center text-sm text-muted-foreground">No paradigms evaluated.</p>
      ) : (
        <ul className="flex flex-col gap-4">
          {sorted.map((paradigm) => (
            <li key={paradigm.name}>
              <div className="flex items-baseline justify-between gap-3">
                <span className="truncate text-sm font-medium" title={paradigm.description}>
                  {humanizeParadigm(paradigm.name)}
                </span>
                <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
                  {formatPercent(paradigm.accuracy)}
                  <span className="ml-2 text-muted-foreground/50">{paradigm.pairs}</span>
                </span>
              </div>
              <DivergingBar accuracy={paradigm.accuracy} />
            </li>
          ))}
        </ul>
      )}

      <p className="mt-6 border-t border-border/60 pt-4 text-[11px] leading-relaxed text-muted-foreground/70">
        Each pair is two sentences differing in one place, one grammatical. The model
        is correct when it assigns the grammatical one a higher total log-probability.
        Chance is 50%; the centre line marks it.
      </p>
    </Panel>
  )
}

function DivergingBar({ accuracy }: { accuracy: number | null }) {
  if (accuracy === null || !Number.isFinite(accuracy)) {
    return (
      <div className="mt-2 h-1.5 w-full rounded-full bg-secondary" aria-label={NO_VALUE} />
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
            ? { left: "50%", width, borderTopRightRadius: 9999, borderBottomRightRadius: 9999 }
            : { right: "50%", width, borderTopLeftRadius: 9999, borderBottomLeftRadius: 9999 }
        }
      />
      {/* The chance line: a hairline tick, not a full rule, so it orients without
          competing with the data. */}
      <div className="absolute left-1/2 top-[-2px] h-[calc(100%+4px)] w-px -translate-x-1/2 bg-border" />
    </div>
  )
}
