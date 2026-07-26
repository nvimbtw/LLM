"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Check } from "lucide-react"

import { Button } from "@/components/ui/button"
import { queueAutolabel, queueGather } from "@/lib/jobs"
import type { SignalStatus } from "@/lib/signals"
import { cn } from "@/lib/utils"

// Sized to clear the catalogue bar in one pass with room to spare: 80 prompts x 6
// candidates is ~480 replies, and the uncertainty band drops some of them. Kept
// deliberately modest — a pass is minutes of GPU time, and a second one appends.
const CATALOGUE_PROMPTS = 80
const CATALOGUE_CANDIDATES = 6

/**
 * The preference cycle, as a checklist with the next step live.
 *
 * **Why a cycle and not a single count.** A preference tune needs thousands of
 * labelled replies; clicking thumbs produces dozens. So the human ratings are not
 * the training data — they are the *seed*, used to check that the rule-based judge
 * agrees with actual taste, after which the judge labels thousands more. Three
 * steps, in order, each unlocking the next:
 *
 *   1. **Rate** a few dozen replies by hand.
 *   2. **Build the catalogue** — the judge is calibrated against those ratings
 *      (and the run refuses if it disagrees with them), then auto-scores hundreds
 *      of generated replies against the same English benchmark the model is graded
 *      on.
 *   3. **Train** on the catalogue.
 *
 * Showing all three at once, rather than only the current blocker, is deliberate:
 * the reason step 1 has a small bar is that it feeds step 2, and that is not
 * guessable from a lone "32 needed".
 */
export interface SignalGateProps {
  /**
   * The count, owned by the form above — which needs `ready` for its own submit
   * button, and fetching it twice would let the two disagree for a render.
   */
  status: SignalStatus | null
  /** The base model being tuned — the one whose replies get rated and labelled. */
  modelId?: string
  disabled?: boolean
}

export function SignalGate({ status, modelId, disabled }: SignalGateProps) {
  const router = useRouter()
  const [isQueueing, setIsQueueing] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  const queue = async (start: () => Promise<{ id_job: string }>) => {
    if (!modelId) return
    setIsQueueing(true)
    setError(null)
    try {
      const job = await start()
      router.push(`/jobs/${job.id_job}`)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to queue the run.")
      setIsQueueing(false)
    }
  }

  if (!status) {
    return (
      <div className="rounded-lg border border-dashed border-border/60 px-4 py-3 text-xs text-muted-foreground">
        Checking the preference catalogue…
      </div>
    )
  }

  return (
    <div className="space-y-3 rounded-lg border p-3">
      <Step
        n={1}
        title="Rate replies by hand"
        done={status.seeded}
        active={!status.seeded}
        count={`${status.total} / ${status.required}`}
        detail={`${status.desirable} good · ${status.undesirable} bad`}
        fraction={status.total / Math.max(1, status.required)}
      >
        <p className="text-xs text-muted-foreground">
          Not the training data — the seed. These are what the auto-scorer gets
          checked against, so it has to agree with you before it labels anything.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            type="button"
            size="sm"
            onClick={() => void queue(() => queueGather({ id_model: modelId! }))}
            disabled={disabled || isQueueing || !modelId}
          >
            {isQueueing ? "Queueing…" : "Gather signals"}
          </Button>
          {status.pending > 0 && (
            <Link
              href="/jobs"
              className="text-xs text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
            >
              {status.pending} generated and waiting to be rated
            </Link>
          )}
        </div>
      </Step>

      <Step
        n={2}
        title="Build the catalogue"
        done={status.catalogue >= status.catalogue_required}
        active={status.seeded && status.catalogue < status.catalogue_required}
        count={`${status.catalogue.toLocaleString()} / ${status.catalogue_required.toLocaleString()}`}
        detail={
          status.catalogue_auto > 0
            ? `${status.catalogue_approved} good · ${status.catalogue_rejected} bad · ${status.catalogue_auto.toLocaleString()} auto-labelled`
            : "nothing auto-labelled yet"
        }
        fraction={status.catalogue / Math.max(1, status.catalogue_required)}
      >
        <p className="text-xs text-muted-foreground">
          The model writes {CATALOGUE_CANDIDATES} replies to each of{" "}
          {CATALOGUE_PROMPTS} prompts, and each one is scored against the same
          English benchmark the model is graded on — n-gram fluency, real words,
          repetition, whether the turn closes. The run stops before writing anything
          if the judge disagrees with your ratings.
        </p>
        <Button
          type="button"
          size="sm"
          onClick={() =>
            void queue(() =>
              queueAutolabel({
                id_model: modelId!,
                candidates: CATALOGUE_CANDIDATES,
                max_prompts: CATALOGUE_PROMPTS,
                temperature: 1.0,
              })
            )
          }
          disabled={disabled || isQueueing || !modelId}
        >
          {isQueueing ? "Queueing…" : "Build catalogue"}
        </Button>
      </Step>

      <Step
        n={3}
        title="Train on it"
        done={false}
        active={status.ready}
        count={status.ready ? `${status.max_epochs} epochs max` : ""}
        detail={
          status.ready
            ? "a fifth of your hand ratings is held back to score the result"
            : ""
        }
      >
        <p className="text-xs text-muted-foreground">
          Ready. Epochs are capped at {status.max_epochs} for a catalogue this size —
          repeated passes over too little data is what narrows a model.
        </p>
      </Step>

      {!modelId && (
        <p className="text-xs text-destructive">
          Pick a base model first; it is the one that writes the replies.
        </p>
      )}
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  )
}

/**
 * One step of the cycle. Collapsed to a line when it is done or not yet reachable,
 * expanded with its explanation and its button when it is the one to do next — so
 * the panel reads as a checklist rather than three competing calls to action.
 */
function Step({
  n,
  title,
  done,
  active,
  count,
  detail,
  fraction,
  children,
}: {
  n: number
  title: string
  done: boolean
  active: boolean
  count: string
  detail?: string
  fraction?: number
  children: React.ReactNode
}) {
  return (
    <div className={cn("space-y-2", !active && !done && "opacity-50")}>
      <div className="flex items-baseline justify-between gap-3">
        <span className="flex items-center gap-2 text-sm">
          <span
            className={cn(
              "flex size-4 shrink-0 items-center justify-center rounded-full border text-[0.6rem] tabular-nums",
              done && "border-transparent bg-foreground text-background"
            )}
          >
            {done ? <Check className="size-2.5" strokeWidth={3} /> : n}
          </span>
          {title}
        </span>
        <span className="text-xs text-muted-foreground tabular-nums">{count}</span>
      </div>

      {fraction !== undefined && (
        <div className="h-1 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-foreground transition-[width] duration-500"
            style={{ width: `${Math.min(100, fraction * 100)}%` }}
          />
        </div>
      )}
      {detail && (
        <p className="text-xs text-muted-foreground tabular-nums">{detail}</p>
      )}

      {active && <div className="space-y-2 pt-0.5">{children}</div>}
    </div>
  )
}
