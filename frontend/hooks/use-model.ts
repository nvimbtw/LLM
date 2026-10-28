"use client"

import * as React from "react"

import { apiFetch, getToken } from "@/lib/api"
import { useSessionToken } from "./use-auth"
import { listDatasets } from "@/lib/data"
import {
  architectureOf,
  baseModelOf,
  datasetsOf,
  mapModel,
  resumedFromOf,
  vocabularySizeOf,
  type LineageDatasetRef,
  type Model,
  type ModelTier,
  type RawArchitecture,
  type RawModel,
  type RawModelConfig,
  type RawModelResult,
} from "./use-models"

export interface ModelDetail extends Model {
  hasWeights: boolean
  createdAt: string
  vocabularySize: number | null
  storedDescription: string
  trainingMode: "pretrain" | "sft" | "kto" | "rl"
  config: RawModelConfig | null
  dataset: RawModelResult | null
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

export interface ModelEdits {
  display_name: string
  description: string
  family: string
  size: ModelTier | null
}

/** Maps resolved model facts alongside the model's own immutable training provenance. */
function mapDetail(raw: RawModel): ModelDetail {
  return {
    ...mapModel(raw),
    hasWeights: raw.has_weights,
    createdAt: raw.created_at ?? "",
    vocabularySize: vocabularySizeOf(raw),
    storedDescription: raw.description ?? "",
    trainingMode:
      raw.parameters?.result?.iterations !== undefined
        ? "rl"
        : (raw.parameters?.config?.mode ??
          (baseModelOf(raw) ? "sft" : "pretrain")),
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

/** Loads one actual-session model and owns its reads/admin writes; replacement aborts transport and cannot publish into another model or account. */
export function useModel(id: string) {
  const token = useSessionToken()
  const resource = `${token}:${id}`
  const [state, setState] = React.useState<{
    resource: string
    model: ModelDetail | null
    notFound: boolean
    error: Error | null
  } | null>(null)
  const lifetime = React.useRef<{
    resource: string
    active: boolean
    controllers: Set<AbortController>
  } | null>(null)
  React.useEffect(() => {
    if (!token || !id) return
    const owner = {
      resource,
      active: true,
      controllers: new Set<AbortController>(),
    }
    lifetime.current = owner
    const controller = new AbortController()
    owner.controllers.add(controller)
    void apiFetch(`/models/${encodeURIComponent(id)}`, {
      signal: controller.signal,
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(async (response) => {
        if (response.status === 404) return null
        if (!response.ok) throw new Error("Unable to load this model.")
        const raw: RawModel = await response.json()
        if (raw.model_id !== id)
          throw new Error("Model response belongs to another resource.")
        return mapDetail(raw)
      })
      .then((model) => {
        if (owner.active)
          setState({ resource, model, notFound: model === null, error: null })
      })
      .catch((cause: unknown) => {
        if (owner.active)
          setState({
            resource,
            model: null,
            notFound: false,
            error:
              cause instanceof Error
                ? cause
                : new Error("Unable to load this model."),
          })
      })
      .finally(() => owner.controllers.delete(controller))
    return () => {
      owner.active = false
      owner.controllers.forEach((current) => current.abort())
      owner.controllers.clear()
    }
  }, [id, token, resource])

  /** Writes explicit admin fields or deletion under this same session/resource, surfacing refusals and suppressing late navigation/snapshot changes. */
  const write = React.useCallback(
    async (
      method: "PATCH" | "DELETE",
      fields?: ModelEdits | { published: boolean }
    ): Promise<void> => {
      const owner = lifetime.current
      if (
        !token ||
        !owner?.active ||
        owner.resource !== resource ||
        getToken() !== token
      )
        throw new Error("Your account or selected model changed. Try again.")
      const controller = new AbortController()
      owner.controllers.add(controller)
      try {
        const response = await apiFetch(`/models/${encodeURIComponent(id)}`, {
          method,
          body: fields ? JSON.stringify(fields) : undefined,
          signal: controller.signal,
          headers: { Authorization: `Bearer ${token}` },
        })
        if (!response.ok)
          throw new Error(
            (await response.text()).slice(0, 2000) ||
              "Unable to update the model."
          )
        if (!owner.active || getToken() !== token)
          throw new Error(
            "Your account or selected model changed. Refresh to check the saved result."
          )
        if (method === "PATCH") {
          const raw: RawModel = await response.json()
          if (raw.model_id !== id)
            throw new Error("Updated model belongs to another resource.")
          if (!owner.active || getToken() !== token)
            throw new Error(
              "Your account or selected model changed. Refresh to check the saved result."
            )
          setState({
            resource,
            model: mapDetail(raw),
            notFound: false,
            error: null,
          })
        } else await response.body?.cancel()
        if (!owner.active || getToken() !== token)
          throw new Error(
            "Your account or selected model changed. Refresh to check the saved result."
          )
      } finally {
        owner.controllers.delete(controller)
      }
    },
    [token, resource, id]
  )

  /** Saves only supplied model display fields for this originating owner. */
  const update = React.useCallback(
    (fields: ModelEdits) => write("PATCH", fields),
    [write]
  )
  /** Changes publication under the actual administrator identity without replacing unsaved source/form state. */
  const setPublished = React.useCallback(
    (published: boolean) => write("PATCH", { published }),
    [write]
  )
  /** Deletes this resource only on explicit action; disposed requests reject before their caller may navigate. */
  const remove = React.useCallback(() => write("DELETE"), [write])
  const current = state?.resource === resource ? state : null
  return {
    model: current?.model ?? null,
    isLoading: Boolean(token && id && !current),
    notFound: current?.notFound ?? false,
    error: current?.error ?? null,
    update,
    setPublished,
    remove,
  }
}

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

/** Returns the construction relationship and ancestor identity for one model response. */
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
 * The chain of models `model` was constructed from. Dataset catalogue failures
 * keep stored names or identities instead of failing the construction history.
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
  const rootNextId = model ? (model.baseModelId ?? model.resumedFromId) : null
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

    /** Reads bounded ancestry until its end, an unavailable model or a cycle, preserving known ancestors. */
    async function walk(): Promise<LineageNode[]> {
      const seen = new Set<string>([rootId as string])
      const out: LineageNode[] = []
      let nextId: string | null = rootNextId
      while (nextId && !seen.has(nextId) && out.length < MAX_LINEAGE_HOPS) {
        seen.add(nextId)
        const res = await apiFetch(`/models/${nextId}`)
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

  React.useEffect(() => {
    let active = true
    listDatasets()
      .then((rows) => {
        if (active) setDatasetNames(new Map(rows.map((d) => [d.id, d.name])))
      })
      .catch(() => {})
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

  /** Resolves a selected dataset's current name, falling back to its stored name and identity. */
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
