"use client"

import { Switch } from "@/components/ui/switch"
import { FormError } from "@/components/ui/error-state"
import { useJobsContext } from "./jobs-provider"
import { AddJobMenu } from "./add-job-menu"
import { ChainBand } from "./chain-band"
import { NowBand } from "./now-band"
import { QueueBand } from "./queue-band"

/**
 * The queue: what is running, what is waiting on the GPU, and what is chained
 * behind a job that has not finished yet.
 *
 * The bands share a single frame rather than sitting in separate cards, because
 * they are one sequence read top to bottom — a job moves down it, and a border
 * between each stage would make that continuity into three lists.
 *
 * Queue above history, always — but no longer *directly* above it. This and
 * `HistoryView` are separate components so the admin panel can put its directory
 * between them: history is the longest thing on that page, and every other admin
 * destination used to sit below a ledger of every run ever made.
 */
export function QueueView() {
  const {
    board,
    running,
    queued,
    waiting,
    paused,
    isLoading,
    error,
    cancel,
    remove,
    reorder,
    pause,
  } = useJobsContext()

  return (
    <div className="space-y-8">
      {error && <FormError>{error}</FormError>}

      <section className="overflow-hidden rounded-(--radius-sheet) border border-border bg-card">
        <header className="flex flex-wrap items-center gap-x-4 gap-y-3 border-b border-border px-5 py-4">
          <div className="min-w-0 flex-1">
            <h2 className="text-sm font-semibold tracking-tight">Queue</h2>
            <p className="mt-0.5 eyebrow text-muted-foreground">
              {summarize(queued.length, running !== null, paused)}
            </p>
          </div>

          {/* Pausing is a state of the ledger, so it lives in its header and dims
              the queued band — not a banner that pushes the page down. It never
              interrupts the job already on the GPU, only stops the next claim. */}
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            <Switch
              checked={paused}
              onCheckedChange={(next) => pause(Boolean(next))}
              aria-label="Pause the queue"
            />
            Pause
          </label>

          <AddJobMenu />
        </header>

        <NowBand job={running} paused={paused} onCancel={cancel} />
        <QueueBand
          jobs={queued}
          paused={paused}
          onReorder={reorder}
          onCancel={cancel}
          onDelete={remove}
        />
        {/* Below the line, because these are not in it: a follow-up starts when the
            job it names finishes, not when its turn comes. */}
        <ChainBand
          jobs={waiting}
          all={[...board.active, ...board.history]}
          onCancel={cancel}
          onDelete={remove}
        />

        {!isLoading &&
          !running &&
          queued.length === 0 &&
          waiting.length === 0 && (
            <p className="px-5 py-8 text-center text-sm text-muted-foreground">
              Nothing queued. Add a job to put the GPU to work.
            </p>
          )}
      </section>
    </div>
  )
}

function summarize(queued: number, busy: boolean, paused: boolean): string {
  if (paused) return queued === 0 ? "Paused" : `Paused · ${queued} waiting`
  if (!busy && queued === 0) return "Idle"
  const waiting = queued === 1 ? "1 waiting" : `${queued} waiting`
  return busy ? `Running · ${waiting}` : waiting
}
