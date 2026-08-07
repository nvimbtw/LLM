"use client"

import * as React from "react"
import { useSearchParams } from "next/navigation"

import {
  FollowUps,
  stepErrors,
  toFollowUps,
  type Step,
} from "@/components/jobs/follow-ups"
import { ParamsRail } from "@/components/training/params-rail"
import { toast } from "@/components/ui/toast"
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
  const { start, isSubmitting } = useTrainingComposer()

  const [cfg, setCfg] = React.useState<TrainConfig>(DEFAULT_CONFIG)
  // What runs after this one, if anything. Held here rather than inside the rail
  // because the rail is one job's parameters and this is a list of other jobs —
  // and because the submit that has to carry both lives here.
  const [steps, setSteps] = React.useState<Step[]>([])

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

  // The rail's own submit is the one submit on the page; the chain rides along with
  // it. A step that does not validate stops the whole submit rather than being
  // silently dropped — a run that queues without the fine-tune you asked to follow
  // it is worse than one that refuses to queue at all, because you find out in the
  // morning.
  const submit = (config: TrainConfig) => {
    const problems = stepErrors(steps)
    if (problems.length > 0) {
      toast.error("Fix the follow-up steps first", problems[0])
      return
    }
    start(config, toFollowUps(steps))
  }

  // Client-side validation stays inside the rail, next to the field it is about.
  // A rejection from the server is an outcome and arrives as a toast.
  return (
    <ParamsRail
      cfg={cfg}
      onChange={setCfg}
      disabled={isSubmitting}
      onStart={submit}
      footer={
        <FollowUps
          steps={steps}
          onChange={setSteps}
          disabled={isSubmitting}
          rootLabel="this run"
        />
      }
    />
  )
}
