import {
  type BenchmarkReport,
  NO_VALUE,
  formatCompact,
  formatNumber,
  formatPercent,
} from "@/lib/benchmark"

// The four numbers worth reading first: the composite, then the one headline
// metric from each stage. Everything else lives in the panels below.
export function ScoreCard({ report, modelName }: { report: BenchmarkReport; modelName: string }) {
  const { corpus, grammar, generation, model } = report

  return (
    <section className="rounded-xl border border-border/60 bg-background">
      <div className="grid gap-px bg-border/60 sm:grid-cols-[minmax(0,1.1fr)_minmax(0,2fr)]">
        {/* The composite, given the space its role deserves. */}
        <div className="bg-background px-6 py-7">
          <div className="text-[10px] font-medium uppercase tracking-[0.16em] text-muted-foreground/60">
            Eval score
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="font-display text-6xl font-semibold tracking-tight tabular-nums">
              {formatNumber(report.score, 1)}
            </span>
            <span className="text-lg text-muted-foreground/50 tabular-nums">/ 100</span>
          </div>
          <p className="mt-3 text-xs text-muted-foreground">
            {modelName} · {model.layers}L · {model.dimensions}d ·{" "}
            {formatCompact(model.parameters)} params
          </p>
        </div>

        {/* gap-px over a border-toned background paints hairline rules between
            the cells — a divided ledger rather than three boxes. */}
        <dl className="grid grid-cols-1 gap-px bg-border/60 sm:grid-cols-3">
          <Headline
            label="Grammar"
            value={formatPercent(grammar.accuracy)}
            hint={`${grammar.pairs} pairs · 50% is chance`}
          />
          <Headline
            label="Bits / byte"
            value={formatNumber(corpus?.bits_per_byte ?? null)}
            hint={
              corpus
                ? `n-gram floor ${formatNumber(corpus.baseline_bits_per_byte)}`
                : "No reference corpus"
            }
          />
          <Headline
            label="Real words"
            value={formatPercent(generation.real_word_rate)}
            hint={
              generation.words_scored > 0
                ? `${generation.words_scored} words generated`
                : "Nothing generated"
            }
          />
        </dl>
      </div>
    </section>
  )
}

function Headline({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="bg-background px-5 py-5">
      <dt className="text-[10px] font-medium uppercase tracking-[0.16em] text-muted-foreground/60">
        {label}
      </dt>
      <dd
        className={`mt-2 font-display text-3xl font-semibold tracking-tight tabular-nums ${
          value === NO_VALUE ? "text-muted-foreground/30" : ""
        }`}
      >
        {value}
      </dd>
      <p className="mt-1 text-[11px] text-muted-foreground/70 tabular-nums">{hint}</p>
    </div>
  )
}
