import { Panel } from "@/components/ui/panel"
import {
  type Sample,
  type SurfaceMetrics,
  formatNumber,
  formatPercent,
} from "@/lib/benchmark"

// What the model actually wrote, and the surface properties of it.
//
// The samples are shown alongside the numbers on purpose: these metrics are the
// ones most likely to be gamed by an artifact (a model that emits one very common
// word forever scores well on real-word rate), and one glance at the text tells
// you whether to believe them.
export function GenerationPanel({
  surface,
  samples,
}: {
  surface: SurfaceMetrics
  samples: Sample[]
}) {
  return (
    <Panel
      title="Generation"
      description={`${samples.length} prompts · ${surface.words} words`}
    >
      <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-border bg-border/60 sm:grid-cols-3 lg:grid-cols-5">
        <Stat
          label="Real words"
          value={formatPercent(surface.real_word_rate)}
          hint="occur in the corpus"
        />
        <Stat
          label="Distinct-2"
          value={formatPercent(surface.distinct_2)}
          hint="unique bigrams"
        />
        <Stat
          label="Distinct-3"
          value={formatPercent(surface.distinct_3)}
          hint="unique trigrams"
        />
        <Stat
          label="Repetition"
          value={formatPercent(surface.repeat_3)}
          hint="recurring trigrams"
        />
        <Stat
          label="Sample"
          value={formatNumber(surface.mean_sample_words, 1)}
          hint="mean words"
        />
      </dl>

      {samples.length > 0 && (
        <ul className="mt-6 flex flex-col gap-px overflow-hidden rounded-lg border border-border bg-border/60">
          {samples.map((sample, i) => (
            <li
              key={i}
              className="bg-background px-4 py-3 text-sm leading-relaxed"
            >
              <span className="text-muted-foreground">{sample.prompt}</span>
              <span className="whitespace-pre-wrap">
                {sample.text || " …"}
              </span>
            </li>
          ))}
        </ul>
      )}

      <p className="mt-4 text-xs leading-relaxed text-muted-foreground">
        Sampled at temperature 0.8, top-k 40, from a fixed seed — the same
        prompts and the same draws on every run, so these numbers are comparable
        across models. The lexicon is built from the reference corpus, so
        &ldquo;real words&rdquo; means &ldquo;strings that occur as words in
        real English text&rdquo;, not &ldquo;in a dictionary&rdquo;.
      </p>
    </Panel>
  )
}

function Stat({
  label,
  value,
  hint,
}: {
  label: string
  value: string
  hint: string
}) {
  return (
    <div className="bg-background px-4 py-3">
      <dt className="text-xs font-medium tracking-[0.14em] text-muted-foreground uppercase">
        {label}
      </dt>
      <dd className="mt-1 font-display text-xl font-semibold tracking-tight tabular-nums">
        {value}
      </dd>
      <p className="text-xs text-muted-foreground">{hint}</p>
    </div>
  )
}
