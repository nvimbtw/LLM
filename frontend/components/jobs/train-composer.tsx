"use client"

import * as React from "react"
import { useSearchParams } from "next/navigation"
import { useSessionToken } from "@/hooks/use-auth"
import { ArchitectureReadOwner } from "@/lib/architecture/read-owner"
import { loadTrainingSeed } from "@/lib/training-seed"

import {
  FollowUps,
  stepErrors,
  toFollowUps,
  type Step,
} from "@/components/jobs/follow-ups"
import { TrainBoard } from "@/components/training/train-board"
import { toast } from "@/components/ui/toast"
import { DEFAULT_CONFIG, useTrainingComposer } from "@/hooks/use-training"
import type { TrainConfig } from "@/lib/training"
import { FormError, LoadError } from "@/components/ui/error-state"
import { Skeleton } from "@/components/ui/skeleton"

/** Keys the private training form by actual bearer and rerun/resume URL selection; replacement releases its draft, read and submission owners. */
export function TrainComposer() {
  const params = useSearchParams()
  const token = useSessionToken()
  const from = params.get("from")
  const resume = params.get("resume")
  return token ? (
    <TrainingForm
      key={`${token}:${from}:${resume}`}
      token={token}
      from={from}
      resume={resume}
    />
  ) : null
}

/** Owns one session and rerun/resume seed; loading finishes before editing starts, and failed seed reads never silently authorize default settings. */
function TrainingForm({
  token,
  from,
  resume,
}: {
  token: string
  from: string | null
  resume: string | null
}) {
  const { start, isSubmitting, submissionError } = useTrainingComposer()
  const [epoch, retrySeed] = React.useReducer((value: number) => value + 1, 0)
  const [seed, setSeed] = React.useState({
    loaded: !from,
    error: null as string | null,
  })

  const [cfg, setCfg] = React.useState<TrainConfig>(() =>
    resume ? { ...DEFAULT_CONFIG, resume_from: resume } : DEFAULT_CONFIG
  )
  const [steps, setSteps] = React.useState<Step[]>([])

  React.useEffect(() => {
    if (!from) return
    const owner = new ArchitectureReadOwner({
      token,
      resource: `training-seed:${from}:${resume}`,
    })
    void owner
      .read((signal) => loadTrainingSeed(from, resume, { token, signal }))
      .then((config) => {
        if (!config) return
        setCfg(config)
        setSeed({ loaded: true, error: null })
      })
      .catch((cause: unknown) =>
        setSeed({
          loaded: false,
          error:
            cause instanceof Error
              ? cause.message.slice(0, 2000)
              : "Previous training settings did not load.",
        })
      )
    return () => owner.dispose()
  }, [token, from, resume, epoch])

  /** Validates every follow-up before requesting one atomic root/chain submission. */
  const submit = (config: TrainConfig) => {
    const problems = stepErrors(steps)
    if (problems.length > 0) {
      toast.error("Fix the follow-up steps first", problems[0])
      return
    }
    start(config, toFollowUps(steps))
  }

  if (seed.error)
    return (
      <LoadError
        title="Previous training settings did not load."
        hint={seed.error}
        onRetry={() => {
          setSeed({ loaded: false, error: null })
          retrySeed()
        }}
      />
    )
  if (!seed.loaded) return <Skeleton className="h-96 w-full" />
  return (
    <>
      {submissionError && <FormError>{submissionError}</FormError>}
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
    </>
  )
}
