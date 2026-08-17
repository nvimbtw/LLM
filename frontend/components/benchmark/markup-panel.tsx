import {
  type MarkupReport,
  NO_VALUE,
  formatNumber,
  formatPercent,
} from "@/lib/benchmark"

/**
 * The markup report, under its score card.
 *
 * Deliberately one plain panel rather than three styled ones. The language bench's
 * panels earned their shape over several iterations of being read; this suite's
 * numbers are new and nobody knows yet which of them will turn out to matter, so
 * they are laid out legibly and left alone until use says otherwise.
 */
export function MarkupPanel({ report }: { report: MarkupReport }) {
  const { positions, structure } = report

  return (
    <div className="flex flex-col gap-6">
      <div className="grid gap-6 xl:grid-cols-2">
        <section className="rounded-lg border border-border bg-background px-6 py-5">
          <h2 className="text-sm font-medium">Marker placement</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            At each marker in a held-out reply: would the model have emitted it?
          </p>
          {positions ? (
            <dl className="mt-5 divide-y divide-border/60">
              <Row
                label="Right marker, whole vocabulary"
                value={formatPercent(positions.marker_top1)}
                hint={`${positions.marker_positions} positions`}
              />
              <Row
                label="Right marker, among markers"
                value={formatPercent(positions.marker_restricted_top1)}
                hint="Ignores whether it knew a marker belonged here"
              />
              <Row
                label="Mean rank of the right marker"
                value={formatNumber(positions.marker_mean_rank, 1)}
                hint="1 is the top choice"
              />
              <Row
                label="Markers in ordinary prose"
                value={formatPercent(positions.false_marker_rate)}
                hint={`${positions.prose_positions} prose positions · lower is better`}
              />
            </dl>
          ) : (
            <Empty />
          )}
        </section>

        <section className="rounded-lg border border-border bg-background px-6 py-5">
          <h2 className="text-sm font-medium">Generated structure</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Replies generated from held-out prompts, checked over the tokens
            themselves.
          </p>
          {structure ? (
            <dl className="mt-5 divide-y divide-border/60">
              <Row
                label="Ended its own turn"
                value={formatPercent(structure.close_rate)}
                hint={`${structure.replies} replies`}
              />
              <Row
                label="Markup balanced and nested"
                value={formatPercent(structure.balance_rate)}
                hint="Every block closed, innermost first"
              />
              <Row
                label="Impersonated another speaker"
                value={formatPercent(structure.scaffold_leak_rate)}
                hint="Emitted <|user|> or <|system|> mid-reply · lower is better"
              />
              <Row
                label="Markers per 100 tokens"
                value={formatNumber(structure.marker_density, 2)}
                hint={`over ${structure.tokens_generated} tokens generated`}
              />
            </dl>
          ) : (
            <Empty />
          )}
        </section>
      </div>
    </div>
  )
}

function Row({
  label,
  value,
  hint,
}: {
  label: string
  value: string
  hint: string
}) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-2.5">
      <dt className="min-w-0">
        <span className="block text-sm">{label}</span>
        <span className="block text-xs text-muted-foreground">{hint}</span>
      </dt>
      <dd
        className={`shrink-0 text-sm tabular-nums ${
          value === NO_VALUE ? "text-muted-foreground/30" : ""
        }`}
      >
        {value}
      </dd>
    </div>
  )
}

// The held-out chat set is written at the start of every benchmark, so this only
// shows before the first run on a fresh install — worth saying, rather than showing
// three dashes and letting it read as a failure.
function Empty() {
  return (
    <p className="mt-5 text-xs text-muted-foreground">
      No held-out chat set yet. It is exported from your SFT conversations when
      a benchmark runs, so this fills in after the next one.
    </p>
  )
}
