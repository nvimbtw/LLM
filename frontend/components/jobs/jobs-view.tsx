"use client"

import { Switch } from "@/components/ui/switch"
import { useJobsContext } from "./jobs-provider"
import { AddJobMenu } from "./add-job-menu"
import { JobHistory } from "./job-history"
import { NowBand } from "./now-band"
import { QueueBand } from "./queue-band"

/**
 * The queue, as one ledger: what is running, what is waiting, what has finished.
 *
 * The three bands share a single frame rather than sitting in separate cards,
 * because they are one sequence read top to bottom — a job moves down it, and a
 * border between each stage would make that continuity into three lists.
 *
 * Queue above history, always. They were briefly side by side to use the width,
 * but a history row carries seven columns of its own — time, label, kind, ran,
 * status, menu — and squeezing that into a third of the page cost more than the
 * empty margin did. The width is better spent letting each row breathe.
 */
export function JobsView() {
  const {
    running,
    runningNetwork,
    queued,
    history,
    paused,
    isLoading,
    error,
    cancel,
    remove,
    retry,
    reorder,
    pause,
  } = useJobsContext()

  return (
    <div className="space-y-8">
      {error && (
        <p className="rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">
          {error}
        </p>
      )}

      <section className="overflow-hidden rounded-lg border border-border bg-card">
        <header className="flex flex-wrap items-center gap-x-4 gap-y-3 border-b border-border px-5 py-4">
          <div className="min-w-0 flex-1">
            <h2 className="text-sm font-semibold tracking-tight">Queue</h2>
            <p className="mt-0.5 text-xs font-medium tracking-[0.12em] text-muted-foreground uppercase">
              {summarize(
                queued.length,
                running !== null || runningNetwork !== null,
                paused
              )}
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
        {/* The network lane gets its own band rather than a slot in the same one:
            it is a second thing happening at the same time, and folding it into
            "what is running" would imply the two are taking turns. */}
        {runningNetwork && (
          <NowBand job={runningNetwork} paused={paused} onCancel={cancel} />
        )}
        <QueueBand
          jobs={queued}
          paused={paused}
          onReorder={reorder}
          onCancel={cancel}
          onDelete={remove}
        />

        {!isLoading && !running && !runningNetwork && queued.length === 0 && (
          <p className="px-5 py-8 text-center text-sm text-muted-foreground">
            Nothing queued. Add a job to put the GPU to work.
          </p>
        )}
      </section>

      <section className="overflow-hidden rounded-lg border border-border bg-card">
        <header className="border-b border-border px-5 py-4">
          <h2 className="text-sm font-semibold tracking-tight">History</h2>
          <p className="mt-0.5 text-xs font-medium tracking-[0.12em] text-muted-foreground uppercase">
            {history.length} {history.length === 1 ? "job" : "jobs"}
          </p>
        </header>
        <JobHistory jobs={history} onRetry={retry} onDelete={remove} />
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
