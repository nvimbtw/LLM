import {
  type AnyReport,
  type AssistantReport,
  type LanguageReport,
  type MarkupReport,
  type ReportModel,
  NO_VALUE,
  isAssistantReport,
  isMarkupReport,
  formatCompact,
  formatNumber,
  formatPercent,
} from "@/lib/benchmark"

// The four numbers worth reading first: the composite, then the one headline
// metric from each stage. Everything else lives in the panels below.
export function ScoreCard({ report }: { report: AnyReport }) {
  if (isMarkupReport(report)) return <MarkupScoreCard report={report} />
  if (isAssistantReport(report)) return <AssistantScoreCard report={report} />
  return <LanguageScoreCard report={report} />
}

/**
 * The composite and what produced it.
 *
 * `model.name` is what a person calls the model; `model.id` is the weights
 * directory. A report from before the name was recorded has only the id, which
 * is a poor label but a true one.
 */
function Composite({
  label,
  score,
  stderr,
  model,
  children,
}: {
  label: string
  score: number | null
  stderr: number | null
  model: ReportModel
  children: React.ReactNode
}) {
  return (
    <section className="rounded-(--radius-sheet) border border-border bg-background">
      <div className="grid gap-px bg-border/60 sm:grid-cols-[minmax(0,1.1fr)_minmax(0,2fr)]">
        <div className="bg-background px-6 py-7">
          <div className="eyebrow text-muted-foreground">
            {label}
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="figure text-6xl">
              {formatNumber(score, 1)}
            </span>
            <span className="text-lg text-muted-foreground tabular-nums">
              / 100
            </span>
          </div>
          <Uncertainty stderr={stderr} />
          <p className="mt-3 text-xs text-muted-foreground">
            {model.name ?? model.id} · {model.layers}L · {model.dimensions}d ·{" "}
            {formatCompact(model.parameters)} params
          </p>
        </div>

        {/* gap-px over a border-toned background paints hairline rules between
            the cells — a divided ledger rather than three boxes. */}
        <dl className="grid grid-cols-1 gap-px bg-border/60 sm:grid-cols-3">
          {children}
        </dl>
      </div>
    </section>
  )
}

function LanguageScoreCard({ report }: { report: LanguageReport }) {
  const { corpus, surface, score, model } = report

  return (
    <Composite
      label="Eval score"
      score={score.score}
      stderr={score.score_stderr}
      model={model}
    >
      <Headline
        label="Floor progress"
        value={formatPercent(corpus?.floor_progress ?? null)}
        hint={corpus ? "of the n-gram floor" : "No reference corpus"}
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
        value={formatPercent(surface?.real_word_rate ?? null)}
        hint={
          surface && surface.words > 0
            ? `${surface.words} words generated`
            : "Nothing generated"
        }
      />
    </Composite>
  )
}

function AssistantScoreCard({ report }: { report: AssistantReport }) {
  const { chat, score, model } = report

  return (
    <Composite
      label="Assistant score"
      score={score.score}
      stderr={score.score_stderr}
      model={model}
    >
      {/* The unit runs the *other* way, so the hint says so rather than leaving
          it to be inferred — the mistake that made this a suite of its own. */}
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
    </Composite>
  )
}

function MarkupScoreCard({ report }: { report: MarkupReport }) {
  const { positions, structure, score, model } = report

  return (
    <Composite
      label="Markup score"
      score={score.score}
      stderr={score.score_stderr}
      model={model}
    >
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
        label="Restraint"
        value={formatPercent(
          positions ? 1 - (positions.false_marker_rate ?? 0) : null
        )}
        hint={
          positions
            ? `over ${positions.prose_positions} prose positions`
            : "No held-out chat set"
        }
      />
      <Headline
        label="Closes turn"
        value={formatPercent(structure?.close_rate ?? null)}
        hint={
          structure
            ? `${structure.replies} replies generated`
            : "Not scored yet"
        }
      />
    </Composite>
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
      <dt className="eyebrow text-muted-foreground">
        {label}
      </dt>
      <dd
        className={`figure mt-2 text-3xl ${
          value === NO_VALUE ? "text-muted-foreground/30" : ""
        }`}
      >
        {value}
      </dd>
      <p className="mt-1 text-xs text-muted-foreground tabular-nums">{hint}</p>
    </div>
  )
}

/**
 * The score's own sampling error, shown beside it.
 *
 * The benchmark is deterministic, so this is not run-to-run variance — it is how
 * far the score would move if the suite drew a different sample of the same size.
 * A delta smaller than this is the sample talking, not the model.
 */
function Uncertainty({ stderr }: { stderr: number | null }) {
  if (stderr === null || !Number.isFinite(stderr)) return null
  return (
    <p className="mt-1 text-xs text-muted-foreground tabular-nums">
      ± {formatNumber(stderr, 2)} at one standard error
    </p>
  )
}
