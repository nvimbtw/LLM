"use client"

import { Switch } from "@/components/ui/switch"
import { AdminSectionHeader } from "@/components/admin/admin-section-header"
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
 * The bands share one open ledger because they are one sequence read top to
 * bottom. The running job carries the emphasis; rules separate the waiting
 * line from follow-ups without turning them into separate cards.
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
    error,
    cancel,
    remove,
    reorder,
    pause,
  } = useJobsContext()

  return (
    <div className="space-y-8">
      {error && <FormError>{error}</FormError>}

      <section>
        <AdminSectionHeader
          title="Queue"
          meta={summarize(queued.length, running !== null, paused)}
          action={
            <>
              {/* Pausing holds the next claim; the running job carries on. */}
              <label className="flex items-center gap-2 text-xs text-muted-foreground">
                <Switch
                  checked={paused}
                  onCheckedChange={(next) => pause(Boolean(next))}
                  aria-label="Pause the queue"
                />
                Pause
              </label>
              <AddJobMenu />
            </>
          }
        />

        <NowBand job={running} onCancel={cancel} />
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
