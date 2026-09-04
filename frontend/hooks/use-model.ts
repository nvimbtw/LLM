"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"
import { listDatasets } from "@/lib/data"
import {
  architectureOf,
  baseModelOf,
  datasetsOf,
  estimateParams,
  mapModel,
  resumedFromOf,
  type LineageDatasetRef,
  type Model,
  type RawArchitecture,
  type RawModel,
  type RawModelConfig,
  type RawModelResult,
} from "./use-models"

// The card-level Model plus the full training config, run summary and timestamps
// the detail page surfaces.
export interface ModelDetail extends Model {
  createdAt: string
  estimatedParams: number
  config: RawModelConfig | null
  dataset: RawModelResult | null
  // Resolved regardless of family — see `architectureOf` for why a pretrain and
  // an SFT run need different reads to answer the same question.
  architecture: RawArchitecture | null
  /** The pretrained model this was fine-tuned from, or null on a pretrain. */
  baseModelId: string | null
  /**
   * The model (finished or a checkpoint) this pretrain continued from, or null
   * on a from-scratch pretrain or an SFT run. Mutually exclusive with
   * `baseModelId` — see `resumedFromOf`.
   */
  resumedFromId: string | null
  /** What this model actually trained on, resolved out of `config`. */
  trainingDatasets: LineageDatasetRef[]
  /**
   * Which of its run's saves this is, and where in the run it was taken — null
   * on an ordinary model. A non-null `checkpointIndex` is the whole definition
   * of "this row is an intermediate", and the page says so rather than
   * presenting a checkpoint as a peer of the model it is an intermediate of.
   */
  checkpointIndex: number | null
  checkpointStep: number | null
  /** The finished model this is an intermediate of, when it is one. */
  parentId: string | null
}

function mapDetail(raw: RawModel): ModelDetail {
  return {
    ...mapModel(raw),
    createdAt: raw.created_at ?? "",
    estimatedParams: estimateParams(raw),
    config: raw.parameters?.config ?? null,
    dataset: raw.parameters?.result ?? null,
    architecture: architectureOf(raw),
    baseModelId: baseModelOf(raw),
    resumedFromId: resumedFromOf(raw),
    trainingDatasets: datasetsOf(raw),
    checkpointIndex: raw.checkpoint_index,
    checkpointStep: raw.checkpoint_step,
    parentId: raw.parent_id,
  }
}

export function useModel(id: string) {
  const [model, setModel] = React.useState<ModelDetail | null>(null)
  const [isLoading, setIsLoading] = React.useState(true)
  const [notFound, setNotFound] = React.useState(false)
  const [error, setError] = React.useState<Error | null>(null)

  // Reset during render rather than in the effect. Switching model ids has to
  // clear the previous model's outcome *before* anything paints, or the page
  // shows the last model's 404 (or its data) for one frame while the new fetch
  // is in flight. React re-runs this render immediately and discards the
  // in-progress output, so nothing downstream ever sees the mismatched pair.
  const [loadedFor, setLoadedFor] = React.useState(id)
  if (loadedFor !== id) {
    setLoadedFor(id)
    setIsLoading(true)
    setNotFound(false)
    setError(null)
  }

  React.useEffect(() => {
    let active = true
    apiFetch(`/models/${id}`)
      .then(async (res) => {
        if (res.status === 404) {
          if (active) setNotFound(true)
          return null
        }
        if (!res.ok) throw new Error("Unable to load this model.")
        return (await res.json()) as RawModel
      })
      .then((raw) => {
        if (active && raw) setModel(mapDetail(raw))
      })
      .catch((err) => {
        if (active)
          setError(
            err instanceof Error ? err : new Error("Unable to load this model.")
          )
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
    }
  }, [id])

  // Admin only. Sends the fields; the backend COALESCEs unchanged ones.
  const update = React.useCallback(
    async (name: string, description: string, family: string) => {
      const res = await apiFetch(`/models/${id}`, {
        method: "PATCH",
        body: JSON.stringify({ display_name: name, description, family }),
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || "Unable to update the model.")
      }
      setModel(mapDetail((await res.json()) as RawModel))
    },
    [id]
  )

  // Admin only. Publish/unpublish toggle.
  const setPublished = React.useCallback(
    async (published: boolean) => {
      const res = await apiFetch(`/models/${id}`, {
        method: "PATCH",
        body: JSON.stringify({ published }),
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || "Unable to update visibility.")
      }
      setModel(mapDetail((await res.json()) as RawModel))
    },
    [id]
  )

  // Admin only. Removes the registry row and the on-disk weights.
  const remove = React.useCallback(async () => {
    const res = await apiFetch(`/models/${id}`, { method: "DELETE" })
    if (!res.ok) {
      const text = await res.text()
      throw new Error(text || "Unable to delete the model.")
    }
  }, [id])

  return { model, isLoading, notFound, error, update, setPublished, remove }
}

// A chain is rarely more than a handful of hops; this is a defensive bound
// against a corrupt or (impossible today, but not enforced by a constraint) a
// cyclic `resume_from`/`base_model`, not a figure anyone is expected to hit.
const MAX_LINEAGE_HOPS = 25

export type LineageRelation = "fine-tuned" | "continued"

/**
 * One model in "how was this constructed" — the current model itself, plus
 * every ancestor its `baseModelId`/`resumedFromId` leads to.
 *
 * A list and not a tree: a model has at most one base and at most one
 * resume-from, never both — a resumed pretrain and a fine-tune are different
 * `family` values — so the construction of a model is a chain, never a
 * branching graph.
 */
export interface LineageNode {
  id: string
  name: string
  family: string
  isRoot: boolean
  /**
   * How this node was built from the *next* one in the chain (its own base or
   * resume-from) — null on the oldest ancestor reached, which names no further
   * lineage of its own.
   */
  relation: LineageRelation | null
  datasets: LineageDatasetRef[]
}

function relationOf(raw: RawModel): {
  relation: LineageRelation | null
  nextId: string | null
} {
  const base = baseModelOf(raw)
  if (base) return { relation: "fine-tuned", nextId: base }
  const resumed = resumedFromOf(raw)
  if (resumed) return { relation: "continued", nextId: resumed }
  return { relation: null, nextId: null }
}

/**
 * The chain of models `model` was constructed from.
 *
 * There is no endpoint that returns a whole ancestry in one call, so this
 * walks it client-side, one `GET /models/{id}` per hop — small in practice,
 * since a chain is rarely more than a few hops, and simpler than adding a
 * server-side walk for traffic this light. `model` is the already-loaded root
 * (from `useModel`); this hook never refetches it, only what it points to.
 *
 * A checkpoint being an intermediate save of a run (`parentId`/
 * `checkpointIndex`) is not a construction edge and never appears here — see
 * `docs/frontend/README.md`'s lineage section for why.
 */
export function useModelLineage(model: ModelDetail | null) {
  const rootId = model?.id ?? null
  const rootNextId = model ? model.baseModelId ?? model.resumedFromId : null
  const rootRelation: LineageRelation | null = model
    ? model.baseModelId
      ? "fine-tuned"
      : model.resumedFromId
        ? "continued"
        : null
    : null

  const [ancestors, setAncestors] = React.useState<LineageNode[]>([])
  const [datasetNames, setDatasetNames] = React.useState<Map<string, string>>(
    new Map()
  )
  const [isLoading, setIsLoading] = React.useState(Boolean(rootNextId))
  const [error, setError] = React.useState<Error | null>(null)
  const [attempt, setAttempt] = React.useState(0)

  // Reset during render, not in the effect — the same shape `useModel` and
  // `useCheckpoints` use: switching to a model with no ancestry (or a
  // different one) has to clear the previous chain before anything paints,
  // and a synchronous `setState` in the effect body is what the
  // `react-hooks/set-state-in-effect` rule (rightly) catches instead.
  const [loadedFor, setLoadedFor] = React.useState(rootNextId)
  if (loadedFor !== rootNextId) {
    setLoadedFor(rootNextId)
    setAncestors([])
    setIsLoading(Boolean(rootNextId))
    setError(null)
  }

  React.useEffect(() => {
    if (!rootId || !rootNextId) return
    let active = true

    async function walk(): Promise<LineageNode[]> {
      const seen = new Set<string>([rootId as string])
      const out: LineageNode[] = []
      let nextId: string | null = rootNextId
      while (nextId && !seen.has(nextId) && out.length < MAX_LINEAGE_HOPS) {
        seen.add(nextId)
        const res = await apiFetch(`/models/${nextId}`)
        // A deleted or inaccessible ancestor ends the chain where it is known
        // rather than failing the whole panel — the models already resolved
        // are still true statements about how this one was built.
        if (!res.ok) break
        const raw = (await res.json()) as RawModel
        const { relation, nextId: pointsTo } = relationOf(raw)
        out.push({
          id: raw.model_id,
          name: raw.name,
          family: raw.family ?? "Custom",
          isRoot: false,
          relation,
          datasets: datasetsOf(raw),
        })
        nextId = pointsTo
      }
      return out
    }

    walk()
      .then((out) => {
        if (active) setAncestors(out)
      })
      .catch((err) => {
        if (active)
          setError(
            err instanceof Error
              ? err
              : new Error("Unable to load this model's lineage.")
          )
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
    }
  }, [rootId, rootNextId, attempt])

  // Dataset names embedded in a stored config can be empty — an older config,
  // or a selection built before one was picked up — so they are resolved
  // against the data engine's own list once, the same "named when we can, by
  // id otherwise" fallback `model-detail.tsx` already uses for a base model.
  React.useEffect(() => {
    let active = true
    listDatasets()
      .then((rows) => {
        if (active) setDatasetNames(new Map(rows.map((d) => [d.id, d.name])))
      })
      .catch(() => {
        // Left empty: nodes fall back to their stored name, then their id.
      })
    return () => {
      active = false
    }
  }, [])

  const nodes: LineageNode[] = React.useMemo(() => {
    if (!model) return []
    const root: LineageNode = {
      id: model.id,
      name: model.name,
      family: model.family,
      isRoot: true,
      relation: rootRelation,
      datasets: model.trainingDatasets,
    }
    return [root, ...ancestors]
  }, [model, rootRelation, ancestors])

  const resolveDatasetName = React.useCallback(
    (dataset: LineageDatasetRef) =>
      datasetNames.get(dataset.id) || dataset.name || dataset.id,
    [datasetNames]
  )

  return {
    nodes,
    resolveDatasetName,
    isLoading,
    error,
    retry: () => {
      setError(null)
      setIsLoading(true)
      setAttempt((n) => n + 1)
    },
  }
}
