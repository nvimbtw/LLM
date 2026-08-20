import { Panel } from "@/components/ui/panel"
import {
  type CorpusMetrics,
  formatCompact,
  formatNumber,
  formatPercent,
} from "@/lib/benchmark"

// Held-out perplexity, always shown against the n-gram floor.
//
// The comparison is the whole point: bits-per-byte has no natural zero, so the
// model's number alone can't be read as good or bad. Two bars on one scale answer
// the only question that matters at this stage — has the transformer beaten a
// counting table fit on the same text?
export function CorpusPanel({ corpus }: { corpus: CorpusMetrics | null }) {
  if (!corpus) {
    return (
      <Panel title="Held-out text" description="Not measured">
        <p className="text-sm text-muted-foreground">
          No reference corpus is installed, so perplexity was skipped.
        </p>
        <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
          Put held-out <code className="font-mono">.txt</code> files in{" "}
          <code className="font-mono">data/eval/corpus/</code> — text the model
          was <em>not</em> trained on.{" "}
          <code className="font-mono">scripts/prepare_eval_corpus.py</code>{" "}
          fills it from the TinyStories train split, which is disjoint from the
          valid split the uploads dir uses.
        </p>
      </Panel>
    )
  }

  const model = corpus.bits_per_byte
  const baseline = corpus.baseline_bits_per_byte
  // Both bars share one scale so their lengths are directly comparable.
  const scale = Math.max(model ?? 0, baseline ?? 0, 0.001)
  const beatsBaseline = model !== null && baseline !== null && model < baseline

  return (
    <Panel
      title="Held-out text"
      description={`${formatCompact(corpus.tokens_scored)} tokens scored`}
    >
      <div className="flex flex-col gap-4">
        <Bar
          label="This model"
          value={model}
          scale={scale}
          className={beatsBaseline ? "bg-primary" : "bg-destructive"}
        />
        <Bar
          label="5-gram floor"
          value={baseline}
          scale={scale}
          className="bg-muted-foreground/40"
        />
      </div>

      <dl className="mt-6 grid grid-cols-2 gap-x-8 gap-y-5 border-y border-border py-4 sm:grid-cols-4">
        <Stat label="Bits / byte" value={formatNumber(model)} />
        <Stat
          label="Perplexity"
          value={formatNumber(corpus.perplexity, 1)}
          hint="per token"
        />
        <Stat
          label="Advantage"
          value={formatPercent(corpus.advantage)}
          hint="over the floor"
        />
        <Stat
          label="Tokens"
          value={formatCompact(corpus.tokens_scored)}
          hint="scored"
        />
      </dl>

      <p className="mt-4 text-xs leading-relaxed text-muted-foreground">
        {beatsBaseline
          ? "Lower is better. The model predicts held-out English more cheaply than an order-5 byte n-gram fit on the same corpus — it has learned something a lookup table has not."
          : "Lower is better. The model does not yet beat an order-5 byte n-gram fit on the same corpus, which means it has not learned structure beyond local byte statistics."}{" "}
        Normalized per byte rather than per token, so the figure stays
        comparable across models with different vocabulary sizes.
      </p>
    </Panel>
  )
}

function Bar({
  label,
  value,
  scale,
  className,
}: {
  label: string
  value: number | null
  scale: number
  className: string
}) {
  const fraction = value !== null && Number.isFinite(value) ? value / scale : 0
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-sm font-medium">{label}</span>
        <span className="text-xs text-muted-foreground tabular-nums">
          {formatNumber(value)}{" "}
          <span className="text-muted-foreground">bits/byte</span>
        </span>
      </div>
      <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className={`h-full rounded-full ${className}`}
          style={{ width: `${Math.max(fraction * 100, 1)}%` }}
        />
      </div>
    </div>
  )
}

function Stat({
  label,
  value,
  hint,
}: {
  label: string
  value: string
  hint?: string
}) {
  return (
    <div className="pr-6">
      <dt className="eyebrow text-muted-foreground">
        {label}
      </dt>
      <dd className="figure mt-1 text-xl">
        {value}
      </dd>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}
