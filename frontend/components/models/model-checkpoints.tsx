"use client"

import Link from "next/link"
import * as React from "react"
import { apiFetch } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { toast } from "@/components/ui/toast"

import { LoadError } from "@/components/ui/error-state"
import { Panel } from "@/components/ui/panel"
import { Skeleton } from "@/components/ui/skeleton"
import { useCheckpoints, type Checkpoint } from "@/hooks/use-checkpoints"
import { formatNumber } from "./format"

/**
 * What the run left behind on its way to this model.
 *
 * Each row is a whole model — its own weights, its own page — so the two things
 * offered are the two things a checkpoint is for: continue training from it, or
 * open it and look at it the way you would any other model. What is deliberately
 * *not* here is the model grid's vocabulary of publishing and deleting: an
 * intermediate is offered as a point in a run, never as a peer of the finished
 * model that lists it.
 *
 * Absent entirely when a run left none — a model trained before checkpointing
 * existed, or one whose run set `checkpoint_target` to 0. An empty panel would
 * read as a section that failed to load.
 */
export interface ModelCheckpointsProps {
  modelId: string
  /** Queueing a run is admin work, so only an admin is offered the continuation. */
  canContinue: boolean
}

/** Lists available recovery models for a parent, with continuation controls only for administrators. */
export function ModelCheckpoints({
  modelId,
  canContinue,
}: ModelCheckpointsProps) {
  const { checkpoints, isLoading, error } = useCheckpoints(modelId)

  if (isLoading) return <Skeleton className="h-44 rounded-(--radius-sheet)" />
  if (error) {
    return (
      <Panel variant="open" title="Checkpoints" bodyClassName="p-0">
        <LoadError title={error.message} className="m-5" />
      </Panel>
    )
  }
  if (checkpoints.length === 0) return null

  const last = checkpoints[checkpoints.length - 1]

  return (
    <Panel
      variant="open"
      title="Recovery and saved checkpoints"
      aside={
        <span className="text-xs text-muted-foreground tabular-nums">
          {checkpoints.length} · to step {formatNumber(last.step)}
        </span>
      }
      bodyClassName="p-0"
    >
      <ul>
        {checkpoints.map((checkpoint) => (
          <CheckpointRow
            key={checkpoint.id}
            checkpoint={checkpoint}
            canContinue={canContinue}
          />
        ))}
      </ul>
    </Panel>
  )
}

/** Presents a saved model’s step and measured loss, with optional retention and continuation actions. */
function CheckpointRow({
  checkpoint,
  canContinue,
}: {
  checkpoint: Checkpoint
  canContinue: boolean
}) {
  const [kept, setKept] = React.useState(checkpoint.kept)
  const [saving, setSaving] = React.useState(false)
  /** Retains this recovery model and reports any persistence failure. */
  const keep = async () => {
    setSaving(true)
    try {
      const response = await apiFetch(`/models/${checkpoint.id}`, {
        method: "PATCH",
        body: JSON.stringify({ recovery_keep: true }),
      })
      if (!response.ok) throw new Error(await response.text())
      setKept(true)
    } catch (error) {
      toast.error(
        "Could not keep checkpoint",
        error instanceof Error ? error.message : "Storage failed"
      )
    } finally {
      setSaving(false)
    }
  }
  return (
    <li className="flex items-baseline gap-4 border-b border-border/40 py-2.5 last:border-0">
      <span className="w-10 shrink-0 text-xs font-medium tabular-nums">
        #{checkpoint.index}
      </span>
      <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground tabular-nums">
        Step {formatNumber(checkpoint.step)}
      </span>

      <span className="w-16 shrink-0 text-right text-xs tabular-nums">
        {checkpoint.loss === null ? (
          <span className="text-muted-foreground/30">—</span>
        ) : (
          checkpoint.loss.toFixed(3)
        )}
      </span>
      <span className="flex shrink-0 items-baseline gap-3 text-xs">
        {canContinue && (
          <Button
            size="sm"
            variant="ghost"
            disabled={kept || saving}
            onClick={keep}
          >
            {kept ? "Kept" : "Keep"}
          </Button>
        )}
        {canContinue && checkpoint.hasWeights && (
          <Link
            href={`/jobs/new-train?resume=${checkpoint.id}`}
            className="text-primary hover:underline"
          >
            Continue
          </Link>
        )}
        <Link
          href={`/models/${checkpoint.id}`}
          className="text-muted-foreground hover:text-foreground hover:underline"
        >
          Inspect
        </Link>
      </span>
    </li>
  )
}
