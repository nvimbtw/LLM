"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"
import type { CurriculumStage, DatasetSelection } from "@/lib/training"

export type ModelTier = "flagship" | "balanced" | "fast"

export interface Model {
  id: string
  name: string
  family: string
  tier: ModelTier
  /** Raw parameter count. 0 means unknown. */
  params: number
  /** Context window in tokens. */
  contextWindow: number
  description: string
  /** Whether the model is published (visible to non-admins). */
  published: boolean
}

// Shape returned by GET /models. `parameters` is the training run's stored
// config and the summary it finished with.
export interface RawModelConfig {
  dimensions?: number
  ctx_window?: number
  heads?: number
  layers?: number
  ff_multiplier?: number
  max_vocab?: number
  lr?: number
  epochs?: number
  // Set only on an SFT run's config — the pretrained model it was fine-tuned
  // from. Absent means this row is a pretrain.
  base_model?: string
  // Set only on a pretrain's config — a model id (finished or a checkpoint) this
  // run continued from. Mutually exclusive with `base_model`: a resumed
  // pretrain and a fine-tune are different `family` values, so a model states
  // at most one kind of ancestry. See `resumedFromOf`.
  resume_from?: string
  // Which datasets this run trained on, and — for a pretrain with staged
  // training — the curriculum they were packed into instead. Present on both
  // families, at the same shape (`lib/training.ts::TrainConfig`/`SftConfig`).
  // See `datasetsOf`.
  datasets?: DatasetSelection[]
  curriculum?: CurriculumStage[]
}

// The five fields that describe a transformer's shape, wherever they came
// from. A *from-scratch* pretrain states these itself (`RawModelConfig`); an
// SFT run and a resumed pretrain state none — both take the shape from the
// checkpoint they start at, and saying nothing is how a resume asks for it —
// so the LLM engine records what it actually built under `result.architecture`
// instead. See `architectureOf`.
export interface RawArchitecture {
  dimensions: number
  heads: number
  layers: number
  ff_multiplier: number
  ctx_window: number
}

// What one intermediate save recorded about itself, as the LLM engine's
// `CheckpointWriter` writes it — the run position it was taken at, and the loss
// there. Every field is optional because it is only present on a checkpoint, and
// only for a checkpoint whose row carries its `parameters` at all.
export interface RawCheckpointResult {
  index?: number
  step?: number
  total_steps?: number
  epoch?: number
  stage?: number
  loss?: number | null
}

export interface RawModelResult {
  files?: number
  chars?: number
  vocab_size?: number
  merges?: number
  corpus_len?: number
  checkpoint?: RawCheckpointResult
  // What the run built, as opposed to what it asked for: an SFT run, a resumed
  // pretrain and every checkpoint row carry this — see `RawArchitecture`.
  architecture?: RawArchitecture
  // Mirrors `RawModelConfig.base_model`; present in both places on an SFT run
  // since the LLM engine writes `result` as the config plus its provenance.
  base_model?: string
}

export interface RawModel {
  model_id: string
  name: string
  description: string | null
  parameters: {
    config?: RawModelConfig
    result?: RawModelResult
  } | null
  context: number | null
  family: string | null
  published: boolean
  has_weights: boolean
  created_at: string | null
  // The run that produced this model, and — on an intermediate save — the
  // finished model it is an intermediate of and where in the run it was taken.
  // `checkpoint_index` being non-null is the whole definition of "this is a
  // checkpoint"; all four are null on an ordinary model.
  run_id: string | null
  parent_id: string | null
  checkpoint_step: number | null
  checkpoint_index: number | null
}

// The model's shape, however its run came by one: stated on the config by a
// from-scratch pretrain, and recorded under `result.architecture` by everything
// that inherits its shape rather than choosing it — an SFT run, a resumed
// pretrain, and the checkpoints either leaves behind (see `RawArchitecture`).
// Every caller that wants to render dimensions/heads/layers/ff_multiplier/
// ctx_window goes through this rather than reading `parameters.config` and
// silently rendering nothing for half the models in the list.
export function architectureOf(raw: RawModel): RawArchitecture | null {
  const c = raw.parameters?.config
  if (c?.dimensions && c.heads && c.layers && c.ff_multiplier && c.ctx_window) {
    return {
      dimensions: c.dimensions,
      heads: c.heads,
      layers: c.layers,
      ff_multiplier: c.ff_multiplier,
      ctx_window: c.ctx_window,
    }
  }
  return raw.parameters?.result?.architecture ?? null
}

// The pretrained model an SFT run started from, or null on a pretrain. Reads
// the config first — where every SFT request names it — falling back to
// `result` for the same reason `architectureOf` does.
export function baseModelOf(raw: RawModel): string | null {
  return raw.parameters?.config?.base_model ?? raw.parameters?.result?.base_model ?? null
}

// The model (finished or a checkpoint) a pretrain continued from, or null on a
// from-scratch pretrain or an SFT run. Unlike `base_model`, `resume_from` has
// no `result` mirror — it names a starting point, not a resolved fact an SFT
// orchestrator derives — so this reads `config` alone.
export function resumedFromOf(raw: RawModel): string | null {
  return raw.parameters?.config?.resume_from ?? null
}

// One dataset a run trained on, resolved to a display name where one is known.
export interface LineageDatasetRef {
  id: string
  /** The name stored with the selection at train time, or "" if none was. */
  name: string
  /** Capped document/conversation count, or null for "all of it". */
  count: number | null
}

// What a model actually trained on, pretrain or SFT alike, flattened out of a
// curriculum's stages when it has one. Order is preserved and a dataset used by
// more than one stage is listed once, at its first appearance — the lineage
// panel shows *what* fed a model, not the schedule it was fed on.
export function datasetsOf(raw: RawModel): LineageDatasetRef[] {
  const config = raw.parameters?.config
  if (!config) return []
  const selections: DatasetSelection[] =
    config.curriculum && config.curriculum.length > 0
      ? config.curriculum.flatMap((stage) => stage.datasets)
      : (config.datasets ?? [])
  const seen = new Set<string>()
  const out: LineageDatasetRef[] = []
  for (const d of selections) {
    if (seen.has(d.id)) continue
    seen.add(d.id)
    out.push({ id: d.id, name: d.name ?? "", count: d.count ?? null })
  }
  return out
}

// Rough transformer parameter count from the resolved architecture — enough
// for a meaningful "size" stat on the card, pretrain or SFT alike. Returns 0
// when no architecture is available.
export function estimateParams(raw: RawModel): number {
  const arch = architectureOf(raw)
  if (!arch) return 0
  const { dimensions: d, layers, ctx_window: ctx, ff_multiplier: ff } = arch
  const vocab = raw.parameters?.result?.vocab_size ?? raw.parameters?.config?.max_vocab ?? 0
  // embeddings (token in + out) + positions + per-layer attention & FFN
  const perLayer = d * d * (4 + 2 * ff)
  return Math.round(2 * vocab * d + ctx * d + layers * perLayer)
}

// Bucket by parameter count: flagship > 100M, balanced > 50M, fast below.
export function tierFor(raw: RawModel): ModelTier {
  const params = estimateParams(raw)
  if (params >= 100_000_000) return "flagship"
  if (params >= 50_000_000) return "balanced"
  return "fast"
}

export function mapModel(raw: RawModel): Model {
  const arch = architectureOf(raw)
  const description =
    raw.description ??
    (arch ? `${arch.layers}-layer transformer, ${arch.dimensions}d` : "Custom trained model")
  return {
    id: raw.model_id,
    name: raw.name,
    family: raw.family ?? "Custom",
    tier: tierFor(raw),
    params: estimateParams(raw),
    contextWindow: raw.context ?? 0,
    description,
    published: raw.published,
  }
}

/**
 * Removes one model: its row, its weights on disk, and the `id_model` on any
 * message that referenced it (detached, not deleted).
 *
 * Admin-only on the backend. Exported as a bare function rather than living on
 * `useModel` because the registry deletes models it is not holding the detail of —
 * see `ModelsGrid`'s batch flow, which needs to delete by id and nothing else.
 */
export async function deleteModel(id: string): Promise<void> {
  const res = await apiFetch(`/models/${id}`, { method: "DELETE" })
  if (!res.ok) {
    throw new Error((await res.text()) || "Failed to delete the model.")
  }
}

// `refreshKey` re-fetches the model list whenever it changes — pass a value that
// advances when a new model is registered (e.g. after a training run finishes) so
// the just-trained model appears without a page reload.
//
// This is the only place the model list is fetched, and it asks for `/models`
// bare. That matters: the LLM engine leaves a run's ~10 intermediate checkpoints out
// unless a request sets `checkpoints=true`, and nothing here ever should — one
// run would otherwise outnumber every finished model in the grid, the chat
// picker and the benchmark composer at once. A checkpoint is offered from the
// one place it belongs to, its parent's page (`useCheckpoints`).
export function useModels(refreshKey?: unknown) {
  const [models, setModels] = React.useState<Model[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<Error | null>(null)

  // See `use-model.ts` — the same reset-during-render, keyed on `refreshKey`, so
  // an advancing key puts the list back into its loading state without a
  // cascading render out of the effect.
  const [loadedFor, setLoadedFor] = React.useState(refreshKey)
  if (loadedFor !== refreshKey) {
    setLoadedFor(refreshKey)
    setIsLoading(true)
  }

  React.useEffect(() => {
    let active = true
    apiFetch(`/models`)
      .then((res) => {
        if (!res.ok) throw new Error("Unable to load models.")
        return res.json() as Promise<{ models: RawModel[] }>
      })
      .then((data) => {
        // A directory with no row has no id to address, so it is not offered.
        if (active)
          setModels(data.models.filter((m) => m.parameters).map(mapModel))
      })
      .catch((err) => {
        if (active)
          setError(
            err instanceof Error ? err : new Error("Unable to load models.")
          )
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
    }
  }, [refreshKey])

  return { models, isLoading, error }
}
