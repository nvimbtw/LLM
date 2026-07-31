"use client"

import Link from "next/link"

import { Panel } from "@/components/ui/panel"
import type { useJob } from "@/hooks/use-job"
import {
  isRunning,
  TIER_LABEL,
  type Job,
  type TeachAccepted,
  type TeachConfig,
  type Tier,
} from "@/lib/jobs"

const TIERS: Tier[] = ["low", "med", "high"]

/**
 * A local-teacher run: how many triples it collected, how they split across the
 * three budgets, and what the teacher actually wrote.
 *
 * The split is the headline and the feed is the check on it. Three roughly equal
 * counts mean the budgets separated; one bucket holding everything means the
 * teacher ignored the instruction and the data teaches nothing about length. Only
 * reading the answers tells you the remaining thing that matters — whether the
 * short replies are as *good* as the long ones, or merely shorter.
 */
export interface JobTeachPanelProps {
  job: Job
  run: ReturnType<typeof useJob>
}

export function JobTeachPanel({ job, run }: JobTeachPanelProps) {
  const config = job.config as TeachConfig | null
  const report = run.teach
  const running = isRunning(job)
  const perTier = report?.per_tier ?? run.teachPerTier
  const answers = perTier.reduce((n, [, count]) => n + count, 0)

  return (
    <>
      {(run.teachCalls.length > 0 || running) && (
        <Panel
          title="While it runs"
          description={running ? (run.stage ?? "Running") : "as it ran"}
        >
          <TierSplit perTier={perTier} />
          <TripleFeed accepted={run.teachAccepted} />
        </Panel>
      )}

      <Panel
        title="Teaching"
        description={config?.teacher?.model_file ?? undefined}
      >
        {report ? (
          <dl className="space-y-2">
            <Row
              label="Triples collected"
              value={report.triples.toLocaleString()}
              hint={`${answers.toLocaleString()} answers across the three files, in ${report.calls} model call${report.calls === 1 ? "" : "s"}.`}
            />
            <Row
              label="Relabelled"
              value={report.relabelled.toLocaleString()}
              hint="Answers filed under a different tier than was asked for. Not an error — the label follows the measurement — but a high number means the word targets in the prompt are mistuned for this teacher."
            />
            <Row
              label="Duplicates skipped"
              value={report.duplicates.toLocaleString()}
              hint="Same opening question as one already in the corpus. A high number means the teacher is circling a few ideas rather than exploring."
            />
            {report.rejected.length > 0 && (
              <Row
                label="Discarded"
                value={report.rejected
                  .reduce((n, [, count]) => n + count, 0)
                  .toLocaleString()}
                hint={report.rejected
                  .map(([reason, count]) => `${count}× ${reason}`)
                  .join(" · ")}
              />
            )}
          </dl>
        ) : running ? (
          <p className="text-sm text-muted-foreground">
            {run.teachTriples > 0
              ? `${run.teachTriples.toLocaleString()} of ${config?.count ?? "?"} triples so far.`
              : "Loading the model onto the GPU…"}
          </p>
        ) : (
          <p className="text-sm text-muted-foreground">
            {job.error ?? "This run produced no report."}
          </p>
        )}

        {report?.gpu === false && (
          <p className="mt-4 rounded-lg border border-dashed border-border px-4 py-3 text-sm text-muted-foreground">
            This build could not offload to the GPU, so the run happened on the
            CPU. The output is no less valid — just produced at a small fraction
            of the speed.
          </p>
        )}

        {report?.measured === false && (
          <p className="mt-4 rounded-lg border border-dashed border-border px-4 py-3 text-sm text-muted-foreground">
            No target model was named, so nothing measured the thinking blocks.
            Every answer is filed under the tier it was <em>asked</em> for
            rather than the one it landed in — check the files on the{" "}
            <Link href="/synth" className="underline underline-offset-4">
              corpus page
            </Link>{" "}
            before trusting the labels.
          </p>
        )}
      </Panel>
    </>
  )
}

/**
 * How the answers divided across the three budgets.
 *
 * Bars rather than numbers alone: the question is not "how many" but "are these
 * three comparable", and a length comparison is what a bar chart is for.
 */
function TierSplit({ perTier }: { perTier: [string, number][] }) {
  const counts = new Map(perTier)
  const total = perTier.reduce((n, [, count]) => n + count, 0)
  if (total === 0) return null
  const max = Math.max(...perTier.map(([, count]) => count), 1)

  return (
    <div className="space-y-2">
      {TIERS.map((tier) => {
        const count = counts.get(tier) ?? 0
        return (
          <div key={tier} className="flex items-center gap-3">
            <span className="w-16 shrink-0 text-xs text-muted-foreground">
              {TIER_LABEL[tier]}
            </span>
            <div className="h-2 min-w-0 flex-1 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-kind-teach"
                style={{ width: `${(count / max) * 100}%` }}
              />
            </div>
            <span className="w-20 shrink-0 text-right text-xs text-muted-foreground tabular-nums">
              {count.toLocaleString()}
              <span className="ml-1 opacity-60">
                {total > 0 ? `${Math.round((count / total) * 100)}%` : ""}
              </span>
            </span>
          </div>
        )
      })}
    </div>
  )
}

/**
 * The most recent triples, newest first — all three answers to each question
 * together, which is the only arrangement that shows whether the tiers differ by
 * reasoning depth or by answer quality.
 */
function TripleFeed({ accepted }: { accepted: TeachAccepted[] }) {
  if (accepted.length === 0) return null

  return (
    <ul className="mt-6 space-y-5 border-t border-border pt-5">
      {accepted.map((triple) => (
        <li key={triple.group} className="space-y-2">
          <p className="line-clamp-2 text-sm">{triple.opener}</p>
          <ul className="space-y-1.5">
            {triple.answers.map((answer, i) => (
              <li key={`${triple.group}-${i}`} className="flex gap-3">
                <span className="w-16 shrink-0 text-xs text-muted-foreground">
                  {TIER_LABEL[answer.filed_as]}
                  {answer.filed_as !== answer.requested && (
                    // Worth marking inline: a feed full of these is the signal
                    // that the prompt's word targets need moving.
                    <span
                      className="ml-1 opacity-60"
                      title={`asked for ${TIER_LABEL[answer.requested]}`}
                    >
                      ↩
                    </span>
                  )}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="line-clamp-2 block text-xs text-muted-foreground">
                    {answer.reply_preview}
                  </span>
                  <span className="text-xs text-muted-foreground tabular-nums opacity-70">
                    {answer.think_tokens.toLocaleString()} thinking tokens
                  </span>
                </span>
              </li>
            ))}
          </ul>
        </li>
      ))}
    </ul>
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
    <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-1">
      <div className="min-w-0 flex-1">
        <dt className="text-sm">{label}</dt>
        <dd className="mt-0.5 text-xs text-muted-foreground">{hint}</dd>
      </div>
      <span className="font-display text-lg font-semibold tabular-nums">
        {value}
      </span>
    </div>
  )
}
