"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"
import type { Model, RawModel } from "./use-models"

export interface CheckpointParent {
  /**
   * The whole model the id belongs to: itself when it names one, its parent
   * when it names a checkpoint, and empty when it names neither.
   */
  parentId: string
  /** The id names an intermediate rather than a whole model. */
  isCheckpoint: boolean
  /**
   * The answer for this id is not in yet. Until it is, nothing may conclude the
   * id is stale — a checkpoint is legitimately absent from every model list, so
   * "not in the list" and "does not exist" look identical right up to here.
   */
  isResolving: boolean
  /**
   * Record a pairing already known for certain — the moment one is clicked.
   * Saves the request that would otherwise re-learn what was just chosen.
   */
  record: (id: string, parent: string) => void
}

/**
 * Which model an id belongs to, when the id may name one of that model's
 * checkpoints.
 *
 * Intermediates are deliberately absent from `useModels` — a run leaves ~10 and
 * they would outnumber the finished models in every list in the product — so an
 * id the list does not hold is either a checkpoint or gone, and only one request
 * tells the two apart. The answer is keyed by the id it was resolved for, so a
 * stale answer can never be read against a different one, and it is written on
 * failure too: the alternative is asking again on every render.
 *
 * Shared by the two places a checkpoint can be chosen — the composer's model
 * picker and the trainer's `ResumePicker` — because both face the same question
 * and neither should answer it differently.
 */
export function useCheckpointParent(
  value: string | undefined,
  models: Model[]
): CheckpointParent {
  const [resolved, setResolved] = React.useState<{
    id: string
    parent: string
  } | null>(null)

  const namesAModel = Boolean(value) && models.some((m) => m.id === value)
  const answered = resolved?.id === value

  React.useEffect(() => {
    if (!value || namesAModel || answered) return
    // An empty list is "not loaded yet", not "no such model" — asking now would
    // spend a request to learn nothing.
    if (models.length === 0) return
    let active = true
    apiFetch(`/models/${value}`)
      .then((res) => (res.ok ? (res.json() as Promise<RawModel>) : null))
      .then((raw) => {
        if (active) setResolved({ id: value, parent: raw?.parent_id ?? "" })
      })
      .catch(() => {
        // An id that resolves to nothing leaves the caller with no parent. What
        // to do about that is the caller's — the composer replaces it, the
        // trainer keeps it and lets the engine refuse the run.
        if (active) setResolved({ id: value, parent: "" })
      })
    return () => {
      active = false
    }
  }, [value, namesAModel, answered, models])

  // The answer for *this* id, or none — an answer held for a different id is
  // not an answer, which is the whole reason it is stored with the id it is for.
  const answer = answered ? resolved : null

  return {
    parentId: !value ? "" : namesAModel ? value : (answer?.parent ?? ""),
    isCheckpoint: Boolean(value) && !namesAModel && Boolean(answer?.parent),
    isResolving: Boolean(value) && !namesAModel && answer === null,
    record: React.useCallback(
      (id: string, parent: string) => setResolved({ id, parent }),
      []
    ),
  }
}
