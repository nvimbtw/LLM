import {
  type AnyReport,
  type AssistantReport,
  type MarkupReport,
  type BenchmarkReport,
  NO_VALUE,
  isAssistantReport,
  isMarkupReport,
  formatCompact,
  formatNumber,
  formatPercent,
} from "@/lib/benchmark"

// The four numbers worth reading first: the composite, then the one headline
// metric from each stage. Everything else lives in the panels below.
export function ScoreCard({
  report,
  modelName,
}: {
  report: AnyReport
  modelName: string
}) {
  if (isMarkupReport(report)) {
    return <MarkupScoreCard report={report} modelName={modelName} />
  }
  if (isAssistantReport(report)) {
    return <AssistantScoreCard report={report} modelName={modelName} />
  }
  return <LanguageScoreCard report={report} modelName={modelName} />
}

function AssistantScoreCard({
  report,
  modelName,
}: {
  report: AssistantReport
  modelName: string
}) {
  const { chat, model } = report

  return (
    <section className="rounded-lg border border-border bg-background">
      <div className="grid gap-px bg-border/60 sm:grid-cols-[minmax(0,1.1fr)_minmax(0,2fr)]">
        <div className="bg-background px-6 py-7">
          <div className="text-xs font-medium tracking-[0.16em] text-muted-foreground uppercase">
            Assistant score
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="font-display text-6xl font-semibold tracking-tight tabular-nums">
              {formatNumber(report.score, 1)}
            </span>
            <span className="text-lg text-muted-foreground tabular-nums">
              / 100
            </span>
          </div>
          <Uncertainty stderr={report.breakdown?.score_stderr ?? null} />
          <p className="mt-3 text-xs text-muted-foreground">
            {modelName} · {model.layers}L · {model.dimensions}d ·{" "}
            {formatCompact(model.parameters)} params
          </p>
        </div>

        {/* The measurement behind the score, shown next to it because the unit runs
            the *other* way — the hint says so rather than leaving it to be inferred,
            which is the mistake that made this a suite of its own. */}
        <dl className="grid grid-cols-1 gap-px bg-border/60 sm:grid-cols-3">
          <Headline
            label="Reply bits"
            value={formatNumber(chat?.bits_per_byte ?? null, 4)}
            hint="per byte · lower is better"
          />
          <Headline
            label="Conversations"
            value={chat ? String(chat.conversations) : NO_VALUE}
            hint={chat ? "held out from training" : "No held-out chat set"}
          />
          <Headline
            label="Tokens scored"
            value={chat ? formatCompact(chat.tokens_scored) : NO_VALUE}
            hint="assistant turns only"
          />
        </dl>
      </div>
    </section>
  )
}

function LanguageScoreCard({
  report,
  modelName,
}: {
  report: BenchmarkReport
  modelName: string
}) {
  const { corpus, grammar, generation, model } = report

  return (
    <section className="rounded-lg border border-border bg-background">
      <div className="grid gap-px bg-border/60 sm:grid-cols-[minmax(0,1.1fr)_minmax(0,2fr)]">
        {/* The composite, given the space its role deserves. */}
        <div className="bg-background px-6 py-7">
          <div className="text-xs font-medium tracking-[0.16em] text-muted-foreground uppercase">
            Eval score
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="font-display text-6xl font-semibold tracking-tight tabular-nums">
              {formatNumber(report.score, 1)}
            </span>
            <span className="text-lg text-muted-foreground tabular-nums">
              / 100
            </span>
          </div>
          <Uncertainty stderr={report.breakdown?.score_stderr ?? null} />
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

function Headline({
  label,
  value,
  hint,
}: {
  label: string
  value: string
  hint: string
}) {
  return (
    <div className="bg-background px-5 py-5">
      <dt className="text-xs font-medium tracking-[0.16em] text-muted-foreground uppercase">
        {label}
      </dt>
      <dd
        className={`mt-2 font-display text-3xl font-semibold tracking-tight tabular-nums ${
          value === NO_VALUE ? "text-muted-foreground/30" : ""
        }`}
      >
        {value}
      </dd>
      <p className="mt-1 text-xs text-muted-foreground tabular-nums">{hint}</p>
    </div>
  )
}

function MarkupScoreCard({
  report,
  modelName,
}: {
  report: MarkupReport
  modelName: string
}) {
  const { pairs, positions, structure } = report.markup
  const { model } = report

  return (
    <section className="rounded-lg border border-border bg-background">
      <div className="grid gap-px bg-border/60 sm:grid-cols-[minmax(0,1.1fr)_minmax(0,2fr)]">
        <div className="bg-background px-6 py-7">
          <div className="text-xs font-medium tracking-[0.16em] text-muted-foreground uppercase">
            Markup score
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="font-display text-6xl font-semibold tracking-tight tabular-nums">
              {formatNumber(report.score, 1)}
            </span>
            <span className="text-lg text-muted-foreground tabular-nums">
              / 100
            </span>
          </div>
          <Uncertainty stderr={report.breakdown?.score_stderr ?? null} />
          <p className="mt-3 text-xs text-muted-foreground">
            {modelName} · {model.layers}L · {model.dimensions}d ·{" "}
            {formatCompact(model.parameters)} params
          </p>
        </div>

        <dl className="grid grid-cols-1 gap-px bg-border/60 sm:grid-cols-3">
          <Headline
            label="Marker pairs"
            value={formatPercent(pairs.accuracy)}
            hint={`${pairs.pairs} pairs · 50% is chance`}
          />
          <Headline
            label="Right marker"
            value={formatPercent(positions?.marker_top1 ?? null)}
            hint={
              positions
                ? `${positions.marker_positions} marker positions`
                : "No held-out chat set"
            }
          />
          <Headline
            label="Closes turn"
            value={formatPercent(structure?.close_rate ?? null)}
            hint={
              structure
                ? `${structure.replies} replies generated`
                : "No held-out chat set"
            }
          />
        </dl>
      </div>
    </section>
  )
}

/**
 * The score's own sampling error, shown beside it.
 *
 * The benchmark is deterministic, so this is not run-to-run variance — it is how far
 * the score would move if the suite drew a different sample of the same size. It is
 * here because the number it qualifies is used to accept or reject post-train rounds,
 * and a delta smaller than this is the pair set talking, not the model.
 */
function Uncertainty({ stderr }: { stderr: number | null }) {
  if (stderr === null || !Number.isFinite(stderr)) return null
  return (
    <p className="mt-1 text-xs text-muted-foreground tabular-nums">
      ± {formatNumber(stderr, 2)} at one standard error
    </p>
  )
}
