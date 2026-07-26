"use client"

import Link from "next/link"

import { CatalogueFigures } from "@/components/jobs/catalogue-figures"
import { verdictColors } from "@/components/training/chart-theme"
import { Panel } from "@/components/ui/panel"
import type { useJob } from "@/hooks/use-job"
import { useSignals } from "@/hooks/use-signals"
import { isRunning, type Job, type Verdict } from "@/lib/jobs"

/**
 * An auto-labelling run: what the judge was measured at, and what it wrote.
 *
 * The calibration line is the one that matters. Everything this job produces is
 * labelled by a rule set rather than a person, so the only reason to trust any of
 * it is that the same rule set was first checked against replies you rated by hand
 * — and the run refuses to write anything if it disagrees with them. Reporting the
 * number keeps that check visible instead of implicit.
 */
export interface JobCataloguePanelProps {
  job: Job
  run: ReturnType<typeof useJob>
}

export function JobCataloguePanel({ job, run }: JobCataloguePanelProps) {
  const report = run.catalogue
  // The live report is one frame at the end of the run and isn't stored on the
  // job, so after a reload the running totals are the honest fallback.
  const { status } = useSignals(report)

  // Live while running, from the final report once it lands — so the number does
  // not vanish when the run ends.
  const calibration = run.calibration ?? report?.calibration ?? null

  return (
    <>
      {(run.verdictScores.length > 0 || isRunning(job)) && (
        <Panel
          title="While it runs"
          description={
            calibration
              ? `judge agrees with your ratings ${(calibration.auc * 100).toFixed(0)}% of the time`
              : "calibrating…"
          }
        >
          <CatalogueFigures
            scores={run.verdictScores}
            threshold={calibration?.threshold}
          />
          <VerdictFeed verdicts={run.verdicts} />
        </Panel>
      )}

      <Panel
        title="Catalogue"
        description={isRunning(job) ? run.stage ?? "Running" : undefined}
      >
      {report ? (
        <>
          <dl className="space-y-2">
            <Figure
              label="Agreement with your ratings"
              value={`${(report.calibration.auc * 100).toFixed(0)}%`}
              hint={`Measured over ${report.calibration.approved} approved and ${report.calibration.rejected} rejected replies you rated. 50% is a coin flip; the run refuses below 65%.`}
            />
            <Figure
              label="Examples written"
              value={report.examples_written.toLocaleString()}
              hint={`${report.approved.toLocaleString()} good · ${report.rejected.toLocaleString()} bad, from ${report.prompts} prompts.`}
            />
            <Figure
              label="Left unlabelled"
              value={report.skipped_uncertain.toLocaleString()}
              hint="Scored too close to the boundary to call. Skipped rather than guessed — a coin-flip label is worse than no label."
            />
          </dl>

          {(report.approved === 0 || report.rejected === 0) && (
            <p className="mt-4 rounded-lg border border-dashed border-border/60 px-4 py-3 text-sm text-muted-foreground">
              Everything landed on one side. A preference tune needs both — it
              learns from the contrast, and with one side empty it just pushes
              every reply the same way. Rate a few more replies so the boundary
              moves, then run this again.
            </p>
          )}
        </>
      ) : (
        <p className="py-6 text-center text-sm text-muted-foreground">
          {job.status === "queued"
            ? "Waiting for the GPU."
            : isRunning(job)
              ? "Calibrating and labelling…"
              : "No report streamed to this page."}
        </p>
      )}

      {status && (
        <div className="mt-4 border-t border-border/40 pt-3 text-sm">
          <div className="flex items-baseline justify-between gap-3">
            <span className="text-muted-foreground">Catalogue now holds</span>
            <span className="tabular-nums">
              {status.catalogue.toLocaleString()} /{" "}
              {status.catalogue_required.toLocaleString()}
            </span>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            {status.ready ? (
              <>
                Enough to tune on.{" "}
                <Link
                  href="/jobs/new/train"
                  className="underline-offset-4 hover:underline"
                >
                  Start the preference run →
                </Link>
              </>
            ) : (
              (status.blocked_reason ?? "")
            )}
          </p>
        </div>
      )}
      </Panel>
    </>
  )
}

/**
 * The replies themselves, newest first, with the verdict each was given.
 *
 * The charts say the catalogue is balanced and separating; only reading the text
 * says whether it is *worth* anything — six hundred well-separated labels over six
 * hundred variations of one bad sentence look identical in aggregate. This is the
 * cheapest possible check against that, and it is why the frames carry the text.
 */
function VerdictFeed({ verdicts }: { verdicts: Verdict[] }) {
  if (verdicts.length === 0) return null

  return (
    <div className="mt-6 space-y-2">
      <p className="text-xs text-muted-foreground">What it generated</p>
      <ul className="max-h-72 space-y-2 overflow-y-auto pr-1">
        {verdicts.map((v, i) => (
          <li
            key={`${v.score}-${i}`}
            className="rounded-md border border-border/50 px-3 py-2 text-sm"
          >
            <div className="flex items-baseline justify-between gap-3">
              <span className="truncate text-xs text-muted-foreground">
                {v.prompt}
              </span>
              <span className="flex shrink-0 items-center gap-1.5 text-xs tabular-nums">
                <span
                  aria-hidden
                  className="inline-block size-2 rounded-[2px]"
                  style={{
                    background: v.good
                      ? verdictColors.approved
                      : verdictColors.rejected,
                  }}
                />
                {/* Stated in words, not colour alone. */}
                {v.good ? "good" : "bad"} · {v.score.toFixed(3)}
              </span>
            </div>
            <p className="mt-1 line-clamp-3 leading-relaxed">{v.response}</p>
          </li>
        ))}
      </ul>
    </div>
  )
}

function Figure({
  label,
  value,
  hint,
}: {
  label: string
  value: string
  hint: string
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-1">
      <div className="min-w-0 flex-1">
        <dt className="text-sm">{label}</dt>
        <dd className="mt-0.5 text-xs text-muted-foreground">{hint}</dd>
      </div>
      <span className="text-sm tabular-nums">{value}</span>
    </div>
  )
}
