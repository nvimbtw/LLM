"use client"

import * as React from "react"
import { useRouter } from "next/navigation"

import { toast } from "@/components/ui/toast"
import { queueTraining } from "@/lib/jobs"
import type { TrainConfig } from "@/lib/training"

export const DEFAULT_CONFIG: TrainConfig = {
  model_name: "my-model",
  dimensions: 64,
  ctx_window: 64,
  heads: 4,
  layers: 2,
  epochs: 5,
  // Pretrain default. Raised from 1e-5 → 1e-4 now that the f32 master-weight fix
  // (ISSUE-0052) lifted the old f16-quantization LR cap: at 1e-5 from-scratch
  // pretrains were badly undertrained. 1e-4 trains stably post-fix (ISSUE-0057).
  lr: 0.0001,
  ff_multiplier: 4,
  max_vocab: 1024,
  datasets: [],
  curriculum: [],
  // Shuffle documents each epoch by default: it isolates documents (an
  // <|endoftext|> separator between them) and removes fixed-order bias at no cost.
  shuffle: true,
  mode: "pretrain",
  // KTO's KL-anchor strength. Carried on the default config (rather than only
  // being set when the Preference mode is picked) so the field is always
  // controlled and the backend never sees an undefined it would have to default.
  beta: 0.1,
}

// ── chart point types ───────────────────────────────────────────────────────
//
// Shared between the live reducer in `use-job` and the figures in
// `components/training`, so they live here rather than beside either one.

// One recorded loss value, tagged with the epoch it belongs to and its position
// (chunk) within that epoch. The chart overlays each epoch as its own line using
// `chunk` as the shared x-axis, so the plot width stays bounded to a single epoch
// instead of growing with the whole run.
export interface LossPoint {
  epoch: number
  chunk: number
  loss: number
}

export interface VocabPoint {
  merges: number
  vocab: number
}

// One report's intra-batch loss spread, plotted over the global step on the
// rolling batch-spread chart.
export interface BatchPoint {
  step: number
  min: number
  avg: number
  max: number
}

/**
 * Composing a training run.
 *
 * Starting a run is now queueing a job: the composer submits, the server puts it
 * in line, and the page hands off to /jobs/{id} where the run's telemetry lives.
 * That split is what lets the composer stay a plain form — it has no live state to
 * own, because by the time there is any, you are on the job's own page.
 */
export function useTrainingComposer() {
  const router = useRouter()
  const [isSubmitting, setIsSubmitting] = React.useState(false)

  const start = React.useCallback(
    async (config: TrainConfig) => {
      setIsSubmitting(true)
      try {
        const job = await queueTraining(config)
        toast.success("Training run queued.", job.label)
        router.push(`/jobs/${job.id_job}`)
      } catch (e) {
        // The backend's 400 names the offending field ("dimensions must be a
        // power of two"), which is the whole value of validating at enqueue —
        // so it is shown as-is rather than replaced with something generic.
        toast.error(
          "Couldn't queue the run",
          e instanceof Error ? e.message : "Something went wrong."
        )
        setIsSubmitting(false)
      }
    },
    [router]
  )

  return { start, isSubmitting }
}
