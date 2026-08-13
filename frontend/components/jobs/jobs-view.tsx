"use client"

import { Switch } from "@/components/ui/switch"
import { useJobsContext } from "./jobs-provider"
import { AddJobMenu } from "./add-job-menu"
import { ChainBand } from "./chain-band"
import { HistoryPager } from "./history-pager"
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
    board,
    running,
    queued,
    waiting,
    history,
    historyTotal,
    historyPage,
    historyPages,
    goToHistoryPage,
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

      <section className="overflow-hidden rounded-lg border border-border bg-card">
        <header className="border-b border-border px-5 py-4">
          <h2 className="text-sm font-semibold tracking-tight">History</h2>
          {/* The count is the whole ledger, not the page — "25 jobs" on every page
              of a hundred would be a worse answer to "how much have I run" than no
              number at all. */}
          <p className="mt-0.5 text-xs font-medium tracking-[0.12em] text-muted-foreground uppercase">
            {historyTotal} {historyTotal === 1 ? "job" : "jobs"}
          </p>
        </header>
        <JobHistory jobs={history} onRetry={retry} onDelete={remove} />
        <HistoryPager
          page={historyPage}
          pages={historyPages}
          onGo={goToHistoryPage}
        />
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
