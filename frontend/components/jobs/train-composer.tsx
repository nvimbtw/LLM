"use client"

import * as React from "react"
import { useSearchParams } from "next/navigation"

import {
  FollowUps,
  stepErrors,
  toFollowUps,
  type Step,
} from "@/components/jobs/follow-ups"
import { TrainBoard } from "@/components/training/train-board"
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
 *
 * `?resume=<model id>` seeds what the run continues from — a finished model or
 * one of its checkpoints, which is the link a model page's checkpoint list
 * carries. Independent of `?from=`: one says what the run *is*, the other says
 * where its weights start, and a run that reuses both is the ordinary way to
 * pick a run back up from the point it went wrong.
 */
export function TrainComposer() {
  const params = useSearchParams()
  const from = params.get("from")
  const resume = params.get("resume")
  const { start, isSubmitting } = useTrainingComposer()

  const [cfg, setCfg] = React.useState<TrainConfig>(() =>
    resume ? { ...DEFAULT_CONFIG, resume_from: resume } : DEFAULT_CONFIG
  )
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
        //
        // `?resume=` wins over whatever the seeded run continued from: the two
        // arrive together when someone re-runs a job *starting from* a
        // checkpoint, and the one in the URL is the one just clicked.
        if (alive && config)
          setCfg(resume ? { ...config, resume_from: resume } : config)
      })
      .catch(() => {
        // A seed that can't be loaded just leaves the defaults in place; the form
        // is still perfectly usable and saying so would be noise.
      })
    return () => {
      alive = false
    }
  }, [from, resume])

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

  // Client-side validation stays inside the board, next to the field it is
  // about. A rejection from the server is an outcome and arrives as a toast.
  return (
    <TrainBoard
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
