"use client"

import * as React from "react"
import { useSearchParams } from "next/navigation"

import { ParamsRail } from "@/components/training/params-rail"
import { DEFAULT_CONFIG, useTrainingComposer } from "@/hooks/use-training"
import { getJob, trainConfig } from "@/lib/jobs"
import type { TrainConfig } from "@/lib/training"

/**
 * The training form.
 *
 * Purely a form: submitting queues a job and hands off to /jobs/{id}, where the
 * run's live figures live — so there is no telemetry state to own here.
 *
 * `?from=<job id>` seeds it from a previous run's stored config. Most runs are a
 * tweak of the last one, and retyping a dozen hyperparameters to change one of
 * them was the worst part of the old flow.
 */
export function TrainComposer() {
  const params = useSearchParams()
  const from = params.get("from")
  const { start, isSubmitting, error } = useTrainingComposer()

  const [cfg, setCfg] = React.useState<TrainConfig>(DEFAULT_CONFIG)

  React.useEffect(() => {
    if (!from) return
    let alive = true
    getJob(from)
      .then((job) => {
        const config = trainConfig(job)
        // The name carries over as-is: training the same name again is a retrain
        // of that model (it reuses the storage id), which is usually what "run
        // again" means. Change it here to fork instead.
        if (alive && config) setCfg(config)
      })
      .catch(() => {
        // A seed that can't be loaded just leaves the defaults in place; the form
        // is still perfectly usable and saying so would be noise.
      })
    return () => {
      alive = false
    }
  }, [from])

  return (
    <>
      {error && (
        <p className="mb-6 rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">
          {error}
        </p>
      )}
      <ParamsRail cfg={cfg} onChange={setCfg} disabled={isSubmitting} onStart={start} />
    </>
  )
}
