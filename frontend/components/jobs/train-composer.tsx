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
import { configForMode } from "@/components/training/run-config"
import type { TrainConfig } from "@/lib/training"
import { FormError, LoadError } from "@/components/ui/error-state"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { useTrainingDraft } from "@/hooks/use-training-draft"
import {
  trainingEntryMode,
  trainingEntryConfig,
  type TrainingEntryMode,
} from "@/lib/training-entry"
import { serializeTrainingDraft } from "@/lib/training-draft"

/** Keys the private training form by actual bearer and rerun/resume URL selection; replacement releases its draft, read and submission owners. */
export function TrainComposer() {
  const params = useSearchParams()
  const token = useSessionToken()
  const draft = useTrainingDraft(token)
  const from = params.get("from")
  const resume = params.get("resume")
  const mode = trainingEntryMode(params.get("mode"))
  if (draft.isLoading) return <Skeleton className="h-96 w-full" />
  if (draft.accountError)
    return (
      <LoadError
        title="Your account could not be loaded for training drafts."
        hint={draft.accountError}
        onRetry={draft.retry}
      />
    )
  return token ? (
    <TrainingForm
      key={`${token}:${from}:${resume}:${mode}`}
      token={token}
      from={from}
      resume={resume}
      mode={mode}
      draft={draft}
    />
  ) : null
}

/** Owns one session and rerun/resume seed; loading finishes before editing starts, and failed seed reads never silently authorize default settings. */
function TrainingForm({
  token,
  from,
  resume,
  mode,
  draft,
}: {
  token: string
  from: string | null
  resume: string | null
  mode: TrainingEntryMode | null
  draft: ReturnType<typeof useTrainingDraft>
}) {
  const {
    draft: savedDraft,
    error: draftError,
    save: saveDraft,
    clear: clearDraft,
  } = draft
  const initialDraft = !from && !resume ? savedDraft : null
  const [epoch, retrySeed] = React.useReducer((value: number) => value + 1, 0)
  const [seed, setSeed] = React.useState({
    loaded: !from,
    error: null as string | null,
  })

  const [cfg, setCfg] = React.useState<TrainConfig>(() =>
    trainingEntryConfig({
      defaults: DEFAULT_CONFIG,
      draft: initialDraft?.config ?? null,
      resume,
      mode,
    })
  )
  const [steps, setSteps] = React.useState<Step[]>(initialDraft?.steps ?? [])
  const [edited, setEdited] = React.useState(Boolean(initialDraft))

  /** Acknowledges only this form's submitted snapshot, preserving a newer draft from another tab. */
  const acknowledgeDraft = () => {
    if (edited) clearDraft(serializeTrainingDraft(cfg, steps))
  }
  const { start, isSubmitting, submissionError } =
    useTrainingComposer(acknowledgeDraft)

  /** Updates root settings and saves the complete draft without requiring valid submission inputs. */
  const changeConfig = (config: TrainConfig) => {
    setCfg(config)
    setEdited(true)
    saveDraft(config, steps)
  }

  /** Updates follow-up order/settings and saves them with the current root configuration. */
  const changeSteps = (next: Step[]) => {
    setSteps(next)
    setEdited(true)
    saveDraft(cfg, next)
  }

  /** Explicitly discards the saved draft and resets the editor to a fresh form in the requested Workbench mode. */
  const discard = () => {
    if (!clearDraft()) return
    setCfg(mode ? configForMode(DEFAULT_CONFIG, mode) : DEFAULT_CONFIG)
    setSteps([])
    setEdited(false)
  }

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
      {draftError && <FormError>{draftError}</FormError>}
      {mode && initialDraft && !from && !resume && cfg.mode !== mode && (
        <div className="mb-6 flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
          <span>
            Restored your{" "}
            {cfg.mode === "pretrain"
              ? "pre-training"
              : cfg.mode === "sft"
                ? "fine-tuning"
                : cfg.mode}{" "}
            draft.
          </span>
          <Button
            variant="text"
            size="sm"
            disabled={isSubmitting}
            onClick={() => changeConfig(configForMode(cfg, mode))}
          >
            Switch to {mode === "sft" ? "fine-tuning" : "pre-training"}
          </Button>
        </div>
      )}
      <TrainBoard
        cfg={cfg}
        onChange={changeConfig}
        disabled={isSubmitting}
        onStart={submit}
        draftStatus={
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            {draftError && <span role="status">Draft not saved</span>}
            {(savedDraft || draftError) && (
              <Button
                type="button"
                variant="text"
                size="sm"
                disabled={isSubmitting}
                onClick={discard}
              >
                Discard draft
              </Button>
            )}
          </div>
        }
        footer={
          <FollowUps
            steps={steps}
            onChange={changeSteps}
            disabled={isSubmitting}
            rootLabel="this run"
          />
        }
      />
    </>
  )
}
