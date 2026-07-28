"use client"

import * as React from "react"
import Link from "next/link"
import { ArrowLeft, Check, Pause, Play, Square } from "lucide-react"

import { CycleFigures } from "@/components/cycles/cycle-figures"
import { Button } from "@/components/ui/button"
import { Panel } from "@/components/ui/panel"
import { RunningDot } from "@/components/ui/running-dot"
import { useCycle } from "@/hooks/use-cycle"
import {
  CYCLE_STATUS_LABEL,
  STAGES,
  STAGE_LABEL,
  roundYield,
  type CycleDetail as Detail,
  type CycleJob,
  type CycleRound,
} from "@/lib/cycles"
import { withParent } from "@/hooks/use-parent-trail"
import { KIND_LABEL, STATUS_LABEL } from "@/lib/jobs"
import { cn } from "@/lib/utils"

/**
 * One cycle, in full — the page that exists so the path does not have to be
 * remembered.
 *
 * It answers, in order: which round is it on and what is it doing right now, is the
 * loop climbing (the yield curve), what did each round decide, and — the one that
 * matters after an unattended night — **why did it stop**.
 */
export interface CycleDetailProps {
  idCycle: string
}

export function CycleDetail({ idCycle }: CycleDetailProps) {
  const { cycle, isLoading, error, setStatus } = useCycle(idCycle)

  if (isLoading) {
    return (
      <p className="py-16 text-center text-sm text-muted-foreground">
        Loading…
      </p>
    )
  }
  if (!cycle) {
    return (
      <p className="py-16 text-center text-sm text-muted-foreground">
        {error ?? "This cycle no longer exists."}
      </p>
    )
  }

  const running = cycle.status === "running"
  const live = running || cycle.status === "paused"

  return (
    <div className="space-y-6">
      <header className="border-b border-border pb-6">
        <Link
          href="/cycles"
          className="mb-4 inline-flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeft className="size-3" />
          Cycles
        </Link>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <h1 className="font-display text-2xl font-semibold tracking-tight">
              {cycle.label}
            </h1>
            <p className="mt-1 flex flex-wrap items-center gap-x-2 text-sm text-muted-foreground">
              <span>{CYCLE_STATUS_LABEL[cycle.status]}</span>
              <span aria-hidden>·</span>
              <span className="tabular-nums">
                round {cycle.round} of {cycle.max_rounds}
              </span>
              {running && (
                <>
                  <span aria-hidden>·</span>
                  <span>{STAGE_LABEL[cycle.stage]}</span>
                </>
              )}
            </p>
          </div>

          <div className="flex items-center gap-2">
            {running && <RunningDot className="mr-1" />}
            {live && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  className="gap-2"
                  onClick={() => void setStatus(running ? "paused" : "running")}
                >
                  {running ? (
                    <Pause className="size-3.5" />
                  ) : (
                    <Play className="size-3.5" />
                  )}
                  {running ? "Pause" : "Resume"}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="gap-2"
                  onClick={() => void setStatus("halted")}
                >
                  <Square className="size-3.5" />
                  Stop
                </Button>
              </>
            )}
          </div>
        </div>
      </header>

      {/* The first thing to read after coming back to it. A halted cycle without a
          visible reason is exactly the "what did this do all night" problem the
          page exists to solve. */}
      {cycle.halt_reason && !live && (
        <Panel
          title="Why it stopped"
          description={CYCLE_STATUS_LABEL[cycle.status]}
        >
          <p className="text-sm">{cycle.halt_reason}</p>
          {cycle.id_model_best && (
            <p className="mt-3 text-sm text-muted-foreground">
              The model to keep is the best round&rsquo;s, not the last
              one&rsquo;s:{" "}
              <Link
                href={`/models/${cycle.id_model_best}`}
                className="underline-offset-4 hover:underline"
              >
                open it
              </Link>
              {cycle.best_score !== null && (
                <span className="tabular-nums">
                  {" "}
                  (scored {cycle.best_score.toFixed(4)})
                </span>
              )}
              .
            </p>
          )}
        </Panel>
      )}

      {running && <StageTrack stage={cycle.stage} />}

      <Panel title="Progress" description="per round">
        <CycleFigures rounds={cycle.rounds} />
      </Panel>

      <Panel title="Rounds" bodyClassName="p-0">
        <RoundTable rounds={cycle.rounds} />
      </Panel>

      <Panel title="Jobs" description="what it queued" bodyClassName="p-0">
        <JobList jobs={cycle.jobs} idCycle={cycle.id_cycle} />
      </Panel>

      <Settings cycle={cycle} />
    </div>
  )
}

/** Where the current round is, as a track rather than a sentence. */
function StageTrack({ stage }: { stage: Detail["stage"] }) {
  const current = STAGES.indexOf(stage)
  return (
    <ol className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-lg border px-4 py-3 text-xs">
      {STAGES.map((s, i) => (
        <li key={s} className="flex items-center gap-2">
          <span
            className={cn(
              "flex items-center gap-1.5",
              i < current && "text-muted-foreground",
              i === current && "font-medium",
              i > current && "text-muted-foreground"
            )}
          >
            {i < current && <Check className="size-3" strokeWidth={3} />}
            {STAGE_LABEL[s]}
          </span>
          {i < STAGES.length - 1 && (
            <span aria-hidden className="text-muted-foreground/30">
              →
            </span>
          )}
        </li>
      ))}
    </ol>
  )
}

function RoundTable({ rounds }: { rounds: CycleRound[] }) {
  if (rounds.length === 0) {
    return (
      <p className="px-5 py-8 text-center text-sm text-muted-foreground">
        No round has started yet.
      </p>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border/40 text-left text-xs text-muted-foreground">
            <Th>Round</Th>
            <Th>Cleared the bar</Th>
            <Th>Added</Th>
            <Th>Score</Th>
            <Th>Verdict</Th>
          </tr>
        </thead>
        <tbody>
          {rounds.map((r) => {
            const y = roundYield(r)
            const added =
              r.approved_after !== null &&
              r.approved_before !== null &&
              r.rejected_after !== null &&
              r.rejected_before !== null
                ? r.approved_after -
                  r.approved_before +
                  (r.rejected_after - r.rejected_before)
                : null
            return (
              <tr
                key={r.id_round}
                className="border-b border-border/20 last:border-0"
              >
                <Td className="tabular-nums">{r.round}</Td>
                <Td className="tabular-nums">
                  {y === null ? "—" : `${(y * 100).toFixed(0)}%`}
                </Td>
                <Td className="text-muted-foreground tabular-nums">
                  {added === null ? "—" : `${added.toLocaleString()} examples`}
                </Td>
                <Td className="tabular-nums">
                  {r.score === null ? "—" : r.score.toFixed(4)}
                </Td>
                <Td>
                  <span className="text-xs">{verdictLabel(r.verdict)}</span>
                  {r.note && (
                    <span className="mt-0.5 block text-xs text-muted-foreground">
                      {r.note}
                    </span>
                  )}
                </Td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function verdictLabel(verdict: string | null): string {
  switch (verdict) {
    case "accepted":
      return "Kept"
    case "no_improvement":
      return "No improvement"
    case "failed":
      return "Failed"
    case null:
      return "In flight"
    default:
      return verdict
  }
}

function JobList({ jobs, idCycle }: { jobs: CycleJob[]; idCycle: string }) {
  if (jobs.length === 0) {
    return (
      <p className="px-5 py-8 text-center text-sm text-muted-foreground">
        Nothing queued yet.
      </p>
    )
  }
  return (
    <ul className="divide-y divide-border/20">
      {jobs.map((job) => (
        <li key={job.id_job}>
          {/* Carrying the cycle means the job's back link returns here rather
              than dumping you in the queue you never came from. */}
          <Link
            href={withParent(`/jobs/${job.id_job}`, `/cycles/${idCycle}`)}
            className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 px-5 py-3 transition-colors hover:bg-muted/40"
          >
            <span className="min-w-0 truncate text-sm">{job.label}</span>
            <span className="flex items-center gap-3 text-xs text-muted-foreground">
              <span>{job.cycle_role ?? KIND_LABEL[job.kind]}</span>
              <span>{STATUS_LABEL[job.status]}</span>
            </span>
          </Link>
        </li>
      ))}
    </ul>
  )
}

/**
 * The knobs the cycle is running under. Shown because they are frozen into the row
 * at creation: two cycles a month apart are only comparable if you can see what
 * each one was told to do.
 */
function Settings({ cycle }: { cycle: Detail }) {
  const c = cycle.config
  const rows: [string, string][] = [
    ["Prompts per round", String(c.max_prompts)],
    ["Candidates per prompt", String(c.candidates)],
    ["Temperature", c.temperature.toFixed(2)],
    ["Self-train LR", c.raft_lr.toExponential(1)],
    ["Preference LR", c.kto_lr.toExponential(1)],
    ["Beta (KL anchor)", c.beta.toFixed(2)],
    ["Min gain to keep a round", c.min_gain.toFixed(4)],
  ]
  return (
    <Panel title="Settings" description="frozen at creation">
      <dl className="space-y-1.5">
        {rows.map(([label, value]) => (
          <div
            key={label}
            className="flex items-baseline justify-between gap-4"
          >
            <dt className="text-sm text-muted-foreground">{label}</dt>
            <dd className="text-sm tabular-nums">{value}</dd>
          </div>
        ))}
      </dl>
    </Panel>
  )
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="px-5 py-2 font-medium">{children}</th>
}

function Td({
  children,
  className,
}: {
  children: React.ReactNode
  className?: string
}) {
  return <td className={cn("px-5 py-3 align-top", className)}>{children}</td>
}
