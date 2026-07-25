import { Panel } from "@/components/ui/panel"
import { type GenerationMetrics, formatNumber, formatPercent } from "@/lib/benchmark"

// What the model actually wrote, and the surface properties of it.
//
// The samples are shown alongside the numbers on purpose: these metrics are the
// ones most likely to be gamed by an artifact (a model that emits one very common
// word forever scores well on real-word rate), and one glance at the text tells
// you whether to believe them.
export function GenerationPanel({ generation }: { generation: GenerationMetrics }) {
  return (
    <Panel
      title="Generation"
      description={`${generation.samples.length} prompts · ${generation.words_scored} words`}
    >
      <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-border/60 bg-border/60 sm:grid-cols-3 lg:grid-cols-5">
        <Stat
          label="Real words"
          value={formatPercent(generation.real_word_rate)}
          hint={`vs ${generation.lexicon_size.toLocaleString("en-US")}-word lexicon`}
        />
        <Stat
          label="Distinct-3"
          value={formatPercent(generation.distinct_3)}
          hint="unique trigrams"
        />
        <Stat
          label="Repetition"
          value={formatPercent(generation.repeat_rate)}
          hint="recurring 4-grams"
        />
        <Stat
          label="Sentence"
          value={formatNumber(generation.mean_sentence_words, 1)}
          hint="mean words"
        />
        <Stat
          label="Markup"
          value={formatPercent(generation.marker_validity)}
          hint="balanced markers"
        />
      </dl>

      {generation.samples.length > 0 && (
        <ul className="mt-6 flex flex-col gap-px overflow-hidden rounded-lg border border-border/60 bg-border/60">
          {generation.samples.map((sample, i) => (
            <li key={i} className="bg-background px-4 py-3 text-sm leading-relaxed">
              <span className="text-muted-foreground/60">{sample.prompt}</span>
              <span className="whitespace-pre-wrap">{sample.completion || " …"}</span>
            </li>
          ))}
        </ul>
      )}

      <p className="mt-4 text-[11px] leading-relaxed text-muted-foreground/70">
        Sampled at temperature 0.8, top-k 40, from a fixed seed — the same prompts and
        the same draws on every run, so these numbers are comparable across models.
        The lexicon is built from the reference corpus, so &ldquo;real words&rdquo;
        means &ldquo;strings that occur as words in real English text&rdquo;, not
        &ldquo;in a dictionary&rdquo;.
      </p>
    </Panel>
  )
}

function Stat({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="bg-background px-4 py-3">
      <dt className="text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground/60">
        {label}
      </dt>
      <dd className="mt-1 font-display text-xl font-semibold tracking-tight tabular-nums">
        {value}
      </dd>
      <p className="text-[10px] text-muted-foreground/60">{hint}</p>
    </div>
  )
}
