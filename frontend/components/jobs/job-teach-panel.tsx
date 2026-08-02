"use client"

import Link from "next/link"

import { Panel } from "@/components/ui/panel"
import { StatRun } from "@/components/training/stat-run"
import type { useJob } from "@/hooks/use-job"
import {
  isRunning,
  TIER_LABEL,
  type Job,
  type TeachAccepted,
  type TeachCall,
  type TeachConfig,
  type TeachReport,
} from "@/lib/jobs"
import { cn } from "@/lib/utils"

import { CoverageBars, RejectionBars, TierDonut } from "./teach-figures"

/**
 * A local-teacher run, in the order the questions get asked.
 *
 * The headline counts first, then the two figures that explain them — what the run
 * produced (the budget split) beside how much of it is usable as a triple — then
 * the diagnostics, then the replies themselves at the bottom, which are the only
 * thing that can answer the question no count can.
 *
 * **Questions, answers and triples are three different units**, and mixing them is
 * what made this panel unreadable before: a question is accepted as soon as *one*
 * of its three answers survives, so "45 triples" beside tier bars summing to 75 was
 * two different measurements wearing one name. Every figure here states its unit.
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
  // The live frame first, then the row. A finished run has no socket to attach to,
  // so without the stored copy every figure below would be empty the moment the run
  // ended — which is exactly how this page used to behave.
  const report = run.teach ?? (job.result as TeachReport | null)
  const running = isRunning(job)

  // A finished run reads its own report; a live one reads the accumulation. The
  // figures take the same shapes either way, so neither has a second code path.
  const perTier = report?.per_tier ?? run.teachPerTier
  const byTiers = report?.questions_by_tiers ?? run.teachByTiers
  const questions = report?.questions ?? run.teachQuestions
  const answers = report?.answers ?? run.teachAnswers
  const complete = report?.complete_triples ?? run.teachComplete
  const calls = report?.calls ?? run.teachCalls.length
  const duplicates = report?.duplicates ?? 0
  const target = config?.count ?? 0

  const hasAnything = questions > 0 || answers > 0 || run.teachCalls.length > 0

  return (
    <>
      <Panel
        title="Teaching"
        description={
          running
            ? (run.stage ?? "Running")
            : (config?.teacher?.model_file ?? "as it ran")
        }
      >
        {hasAnything ? (
          <StatRun
            stats={[
              {
                label: target > 0 ? `Questions of ${target}` : "Questions",
                value: questions.toLocaleString(),
              },
              { label: "Answers written", value: answers.toLocaleString() },
              { label: "Complete triples", value: complete.toLocaleString() },
              { label: "Model calls", value: calls.toLocaleString() },
              {
                label: "Answers per call",
                value: calls > 0 ? (answers / calls).toFixed(1) : "—",
              },
            ]}
          />
        ) : (
          <p className="text-sm text-muted-foreground">
            {running
              ? "Loading the model onto the GPU…"
              : (job.error ?? "This run produced no report.")}
          </p>
        )}

        {report?.gpu === false && (
          <Note>
            This build could not offload to the GPU, so the run happened on the
            CPU. The output is no less valid — just produced at a small fraction
            of the speed.
          </Note>
        )}

        {report?.measured === false && (
          <Note>
            No target model was named, so nothing measured the thinking blocks.
            Every answer is filed under the tier it was <em>asked</em> for
            rather than the one it landed in — check the files on the{" "}
            <Link href="/synth" className="underline underline-offset-4">
              corpus page
            </Link>{" "}
            before trusting the labels.
          </Note>
        )}
      </Panel>

      {hasAnything && (
        // Side by side because they are one thought in two halves: what came out,
        // and how much of it is the thing the format exists for. Stacked on a
        // narrow screen, split-to-whole first.
        // `gap-6` matches the vertical rhythm the panel stack uses, so the space
        // between these two is the same in both directions.
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel
            title="Budget split"
            description="Answers across the three files"
          >
            <TierDonut perTier={perTier} />
          </Panel>

          <Panel
            title="Triple completeness"
            description="Budgets answered per question"
          >
            <CoverageBars byTiers={byTiers} />
          </Panel>
        </div>
      )}

      {hasAnything && (
        <Panel title="What went wrong" description="And how often">
          <dl className="mb-5 space-y-2">
            <Row
              label="Relabelled"
              value={(report?.relabelled ?? 0).toLocaleString()}
              hint="Answers filed under a different budget than was asked for. Not an error — the label follows the measurement — but a high number means the word targets in the prompt are mistuned for this teacher."
            />
            <Row
              label="Duplicates skipped"
              value={duplicates.toLocaleString()}
              hint="Same opening question as one already in the corpus. A high number means the teacher is circling a few ideas rather than exploring."
            />
          </dl>
          <RejectionBars
            rejected={report?.rejected ?? lastRejected(run.teachCalls)}
          />
        </Panel>
      )}

      {run.teachAccepted.length > 0 && (
        <Panel
          title="What it wrote"
          description={`${run.teachAccepted.length} most recent, newest first`}
        >
          <TripleFeed accepted={run.teachAccepted} />
        </Panel>
      )}
    </>
  )
}

// The running tally, for a live run that has no final report yet. Each call frame
// carries the run's cumulative rejections, so the newest frame is the whole story
// — no accumulation needed on this side.
function lastRejected(calls: TeachCall[]): [string, number][] {
  return calls.length > 0 ? calls[calls.length - 1].rejected : []
}

/**
 * The most recent triples, newest first — every answer to a question together,
 * which is the only arrangement that shows whether the budgets differ by reasoning
 * depth or merely by answer quality.
 *
 * The thinking-token count sits on each answer because it is the measurement the
 * filing is based on: a "long" answer with forty thinking tokens is the run's
 * failure mode made visible, and no aggregate figure on this page would show it.
 */
function TripleFeed({ accepted }: { accepted: TeachAccepted[] }) {
  return (
    <ul className="space-y-6">
      {accepted.map((triple) => (
        <li
          key={triple.group}
          className="space-y-3 border-b border-border/50 pb-6 last:border-0 last:pb-0"
        >
          <div className="flex items-baseline justify-between gap-4">
            <p className="min-w-0 flex-1 text-sm font-medium">
              {triple.opener}
            </p>
            {/* A question that got all three is worth marking: they are rare, and
                they are the ones the format is for. */}
            {triple.answers.length === 3 && (
              <span className="shrink-0 rounded-full bg-kind-teach/12 px-2 py-0.5 text-xs text-kind-teach">
                complete
              </span>
            )}
          </div>

          <ul className="space-y-2.5">
            {triple.answers.map((answer, i) => (
              <li key={`${triple.group}-${i}`} className="flex gap-3">
                <span className="flex w-20 shrink-0 items-baseline gap-1.5 pt-0.5">
                  <span
                    aria-hidden
                    className="size-2 shrink-0 translate-y-[-1px] rounded-full"
                    style={{ background: `var(--tier-${answer.filed_as})` }}
                  />
                  <span className="text-xs text-muted-foreground">
                    {TIER_LABEL[answer.filed_as]}
                  </span>
                </span>
                <span className="min-w-0 flex-1">
                  <span className="line-clamp-3 block text-xs leading-relaxed text-muted-foreground">
                    {answer.reply_preview}
                  </span>
                  <span className="mt-1 flex flex-wrap items-center gap-x-3 text-xs text-muted-foreground/70 tabular-nums">
                    <span>
                      {answer.think_tokens.toLocaleString()} thinking tokens
                    </span>
                    {answer.filed_as !== answer.requested && (
                      // Worth naming in words rather than as a symbol: a feed full
                      // of these is the signal that the prompt's word targets need
                      // moving, and an arrow glyph never said which way.
                      <span className="text-muted-foreground/70">
                        asked for {TIER_LABEL[answer.requested].toLowerCase()}
                      </span>
                    )}
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

function Note({ children }: { children: React.ReactNode }) {
  return (
    <p className="mt-4 rounded-lg border border-dashed border-border px-4 py-3 text-sm text-muted-foreground">
      {children}
    </p>
  )
}

function Row({
  label,
  value,
  hint,
  className,
}: {
  label: string
  value: string
  hint: string
  className?: string
}) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-start justify-between gap-x-6 gap-y-1",
        className
      )}
    >
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
