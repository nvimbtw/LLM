"use client"

import Link from "next/link"

import { SynthFigures } from "@/components/jobs/synth-figures"
import { Panel } from "@/components/ui/panel"
import type { useJob } from "@/hooks/use-job"
import {
  isRunning,
  type Job,
  type SynthAccepted,
  type SynthConfig,
} from "@/lib/jobs"

/**
 * A synthetic-data run: how much it added, what it threw away, and what the model
 * on the other end actually wrote.
 *
 * The feed is the part worth watching. Counts cannot tell you that two hundred
 * accepted conversations are the same three questions reworded — only reading them
 * can, and a run is long enough that reading a few while it goes is free.
 */
export interface JobSynthPanelProps {
  job: Job
  run: ReturnType<typeof useJob>
}

export function JobSynthPanel({ job, run }: JobSynthPanelProps) {
  const config = job.config as SynthConfig | null
  const report = run.synth
  const target = report?.added ?? run.synthAdded
  const running = isRunning(job)

  return (
    <>
      {(run.synthCalls.length > 0 || running) && (
        <Panel
          title="While it runs"
          description={running ? (run.stage ?? "Running") : "as it ran"}
        >
          <SynthFigures calls={run.synthCalls} />
          <AcceptedFeed accepted={run.synthAccepted} />
        </Panel>
      )}

      <Panel title="Generation" description={config?.model ?? undefined}>
        {report ? (
          <dl className="space-y-2">
            <Row
              label="Conversations added"
              value={report.added.toLocaleString()}
              hint={`Written to ${report.out_file}, in ${report.calls} model call${report.calls === 1 ? "" : "s"}. The SFT corpus now holds ${report.corpus_total.toLocaleString()}.`}
            />
            <Row
              label="With a thinking block"
              value={`${report.with_thinking.toLocaleString()} of ${report.added.toLocaleString()}`}
              hint="Large reasoning blocks are what generation adds over the hand-written set, which has almost none."
            />
            <Row
              label="Duplicates skipped"
              value={report.duplicates.toLocaleString()}
              hint="Same opening question as a conversation already in the corpus. A high number means the model is circling a few ideas rather than exploring."
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
            {target > 0
              ? `${target.toLocaleString()} of ${config?.count ?? "?"} so far.`
              : "Waiting for the first call to come back…"}
          </p>
        ) : (
          <p className="text-sm text-muted-foreground">
            {job.error ?? "This run produced no report."}
          </p>
        )}

        {config?.id_model == null && (
          <p className="mt-4 rounded-lg border border-dashed border-border px-4 py-3 text-sm text-muted-foreground">
            This run named no target model, so nothing checked whether the
            conversations fit a context window. Check them on the{" "}
            <Link href="/synth" className="underline underline-offset-4">
              corpus page
            </Link>{" "}
            before training on them.
          </p>
        )}
      </Panel>
    </>
  )
}

/**
 * The most recent conversations, newest first. Truncated hard: this is a window
 * onto the run, not a reader — the file on disk is the real artefact.
 */
function AcceptedFeed({ accepted }: { accepted: SynthAccepted[] }) {
  if (accepted.length === 0) return null

  return (
    <ul className="mt-6 space-y-3 border-t border-border pt-5">
      {accepted.map((item, i) => (
        <li key={`${item.opener}-${i}`} className="space-y-1">
          <p className="line-clamp-2 text-sm">{item.opener}</p>
          <p className="line-clamp-2 text-xs text-muted-foreground">
            {item.reply}
          </p>
          <p className="text-xs text-muted-foreground tabular-nums">
            {item.turns} turns
            {item.tokens != null && <> · {item.tokens} tokens</>}
            {item.think_words > 0 && (
              <> · {item.think_words} words of thinking</>
            )}
          </p>
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
