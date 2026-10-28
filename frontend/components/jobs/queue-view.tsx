"use client"

import { Switch } from "@/components/ui/switch"
import { AdminSectionHeader } from "@/components/admin/admin-section-header"
import { FormError } from "@/components/ui/error-state"
import { useJobsContext } from "./jobs-provider"
import { AddJobMenu } from "./add-job-menu"
import { ChainBand } from "./chain-band"
import { NowBand } from "./now-band"
import { QueueBand } from "./queue-band"

/** Shows current work, ordered dispatch and dependent follow-ups; pausing affects only the next claim. */
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
              <label className="flex items-center gap-2 text-xs text-muted-foreground">
                <Switch
                  checked={paused}
                  onCheckedChange={(next) => pause(Boolean(next))}
                  aria-label={paused ? "Resume the queue" : "Pause the queue"}
                />
                {paused ? "Resume queue" : "Pause queue"}
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

/** Summarizes dispatch state and queued count without claiming a running job is paused. */
function summarize(queued: number, busy: boolean, paused: boolean): string {
  if (paused) return queued === 0 ? "Paused" : `Paused · ${queued} waiting`
  if (!busy && queued === 0) return "Idle"
  const waiting = queued === 1 ? "1 waiting" : `${queued} waiting`
  return busy ? `Running · ${waiting}` : waiting
}
