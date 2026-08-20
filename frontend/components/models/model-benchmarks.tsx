"use client"

import Link from "next/link"

import { Skeleton } from "@/components/ui/skeleton"
import { useModelBenchmarks } from "@/hooks/use-model-benchmarks"
import {
  NO_VALUE,
  formatDate,
  formatNumber,
  formatPercent,
  isLanguageReport,
  type BenchmarkRun,
  type LanguageReport,
} from "@/lib/benchmark"
import { cn } from "@/lib/utils"
import { LoadError } from "@/components/ui/error-state"

/**
 * How the model scored, under the specs that say what it is.
 *
 * A summary, not a second copy of /benchmark. The question this section answers is
 * "is this model any good", which is the composite plus the three stage headlines;
 * the paradigm-by-paradigm breakdown and the generated samples are a different
 * question, and they stay on the page built for them — the run list here links
 * straight there.
 *
 * Rendered only for admins, because the benchmark API is admin-only. A model with
 * no runs still renders the section: "never benchmarked" is an answer, and an
 * absent section reads as a page that forgot to load.
 */
export interface ModelBenchmarksProps {
  modelId: string
}

export function ModelBenchmarks({ modelId }: ModelBenchmarksProps) {
  const { data, isLoading, error } = useModelBenchmarks(modelId)

  if (isLoading) return <Skeleton className="h-44 rounded-(--radius-sheet)" />
  if (error) {
    return (
      <Section>
        <LoadError title={error} className="m-5" />
      </Section>
    )
  }

  const latest = data?.latest ?? null
  const runs = data?.runs ?? []

  if (!latest?.report) {
    return (
      <Section>
        <p className="px-5 py-6 text-sm text-muted-foreground">
          {runs.length > 0
            ? "Every run of this model failed before it produced a report."
            : "Never benchmarked."}{" "}
          <Link href="/jobs/new/benchmark" className="underline">
            Queue a run
          </Link>
          .
        </p>
      </Section>
    )
  }

  return (
    <Section subtitle={`Scored ${formatDate(latest.created_at)}`}>
      {/* The model page shows the language bench only — a second and third set of
          four figures beside it would compete for the same glance. The markup and
          assistant scores are a click away on /benchmark. `getModelRuns` defaults to
          the language suite, so no other report can arrive here; the guard is for the
          type, not for a case — and it names the suite it wants rather than excluding
          one, which stopped being sufficient the moment there were three. */}
      {isLanguageReport(latest.report) && <Headlines report={latest.report} />}
      {runs.length > 1 && <RunList runs={runs} />}
    </Section>
  )
}

function Section({
  subtitle,
  children,
}: {
  subtitle?: string
  children: React.ReactNode
}) {
  return (
    <section className="rounded-(--radius-sheet) border border-border bg-background">
      <header className="flex items-baseline justify-between gap-3 border-b border-border px-5 py-3.5">
        <h2 className="eyebrow text-muted-foreground">
          Benchmarks
        </h2>
        {subtitle && (
          <span className="text-xs text-muted-foreground tabular-nums">
            {subtitle}
          </span>
        )}
      </header>
      {children}
    </section>
  )
}

// The composite beside the three stage headlines — the same divided-ledger figure
// the hero strip and /benchmark's score card both use, so a score reads the same
// wherever it is met.
function Headlines({ report }: { report: LanguageReport }) {
  const { corpus, surface, score } = report

  return (
    <dl className="grid grid-cols-2 gap-px bg-border/60 sm:grid-cols-4">
      <Figure
        label="Eval score"
        value={formatNumber(score.score, 1)}
        hint="out of 100"
        lead
      />
      <Figure
        label="Floor progress"
        value={formatPercent(corpus?.floor_progress ?? null)}
        hint={corpus ? "of the n-gram floor" : "No reference corpus"}
      />
      <Figure
        label="Bits / byte"
        value={formatNumber(corpus?.bits_per_byte ?? null)}
        hint={
          corpus
            ? `n-gram floor ${formatNumber(corpus.baseline_bits_per_byte)}`
            : "No reference corpus"
        }
      />
      <Figure
        label="Real words"
        value={formatPercent(surface?.real_word_rate ?? null)}
        hint={
          surface && surface.words > 0
            ? `${surface.words} words generated`
            : "Nothing generated"
        }
      />
    </dl>
  )
}

function Figure({
  label,
  value,
  hint,
  lead,
}: {
  label: string
  value: string
  hint: string
  lead?: boolean
}) {
  return (
    <div className="bg-background px-5 py-4">
      <dt className="eyebrow text-muted-foreground">
        {label}
      </dt>
      <dd
        className={cn(
          "figure mt-1.5",
          lead ? "text-3xl" : "text-2xl",
          value === NO_VALUE && "text-muted-foreground/30"
        )}
      >
        {value}
      </dd>
      <p className="mt-0.5 text-xs text-muted-foreground tabular-nums">
        {hint}
      </p>
    </div>
  )
}

// Earlier runs, so a score reads as a point on a trajectory rather than a fact.
// Newest first, and the newest is the one the figures above describe.
function RunList({ runs }: { runs: BenchmarkRun[] }) {
  return (
    <ul className="border-t border-border/40">
      {runs.map((run) => (
        <li
          key={run.id_run}
          className="flex items-baseline gap-3 border-b border-border/40 px-5 py-2 last:border-0"
        >
          <Link
            href="/benchmark"
            className="text-xs text-muted-foreground tabular-nums hover:underline"
          >
            {formatDate(run.created_at)}
          </Link>
          <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground">
            {run.status === "failed" ? (run.error ?? "Failed") : ""}
          </span>
          <span
            className={cn(
              "text-xs font-medium tabular-nums",
              run.status === "failed" && "text-destructive/70"
            )}
          >
            {run.score === null
              ? run.status === "failed"
                ? "Failed"
                : "—"
              : formatNumber(run.score, 1)}
          </span>
        </li>
      ))}
    </ul>
  )
}
