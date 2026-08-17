import { API_URL, authHeaders } from "./api"
import { DEFAULT_CONFIG } from "@/hooks/use-training"
import type { CurriculumStage, DatasetSelection, TrainConfig } from "./training"

// A named, reusable training-parameter set. `config` is a TrainConfig-shaped blob
// the backend stores opaquely (validated only when a run actually starts). Mirrors
// the response from routes/preset.rs.
export interface Preset {
  id_preset: string
  name: string
  config: TrainConfig
  updated_at: string
}

export async function listPresets(): Promise<Preset[]> {
  const res = await fetch(`${API_URL}/presets`, {
    headers: authHeaders(),
  })
  if (!res.ok) return []
  return (await res.json()) as Preset[]
}

export async function createPreset(
  name: string,
  config: TrainConfig
): Promise<Preset> {
  const res = await fetch(`${API_URL}/presets`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ name, config }),
  })
  if (!res.ok) {
    // 409 = a preset by this name already exists for the user.
    throw new Error(
      res.status === 409
        ? "A preset with that name already exists."
        : (await res.text()) || "Failed to save preset."
    )
  }
  return (await res.json()) as Preset
}

export async function updatePreset(
  id: string,
  patch: { name?: string; config?: TrainConfig }
): Promise<Preset> {
  const res = await fetch(`${API_URL}/presets/${id}`, {
    method: "PUT",
    headers: authHeaders(),
    body: JSON.stringify(patch),
  })
  if (!res.ok) {
    throw new Error(
      res.status === 409
        ? "A preset with that name already exists."
        : (await res.text()) || "Failed to update preset."
    )
  }
  return (await res.json()) as Preset
}

export async function deletePreset(id: string): Promise<void> {
  const res = await fetch(`${API_URL}/presets/${id}`, {
    method: "DELETE",
    headers: authHeaders(),
  })
  if (!res.ok && res.status !== 404) {
    throw new Error("Failed to delete preset.")
  }
}

// Coerce a possibly-untrusted stored value into a finite number, else fall back.
function num(v: unknown, fallback: number): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback
}

// Coerce a stored blob's dataset selection into a clean DatasetSelection[],
// dropping any entry that isn't a named object and clamping its count to a
// non-negative integer (0 = all). Missing/garbage yields an empty selection.
function datasets(v: unknown): DatasetSelection[] {
  if (!Array.isArray(v)) return []
  const out: DatasetSelection[] = []
  for (const item of v) {
    if (!item || typeof item !== "object") continue
    const name = (item as { name?: unknown }).name
    if (typeof name !== "string" || !name.trim()) continue
    const rawCount = (item as { count?: unknown }).count
    const count =
      typeof rawCount === "number" && Number.isFinite(rawCount) && rawCount > 0
        ? Math.floor(rawCount)
        : 0
    out.push({ name, count })
  }
  return out
}

// Coerce a stored blob's curriculum into a clean CurriculumStage[], dropping any
// stage without a valid dataset selection and clamping epochs to a positive
// integer. Missing/garbage yields an empty curriculum (the plain single-pass run).
function curriculum(v: unknown): CurriculumStage[] {
  if (!Array.isArray(v)) return []
  const out: CurriculumStage[] = []
  for (const item of v) {
    if (!item || typeof item !== "object") continue
    const stageDatasets = datasets((item as { datasets?: unknown }).datasets)
    if (stageDatasets.length === 0) continue
    const rawEpochs = (item as { epochs?: unknown }).epochs
    const epochs =
      typeof rawEpochs === "number" &&
      Number.isFinite(rawEpochs) &&
      rawEpochs > 0
        ? Math.floor(rawEpochs)
        : 1
    const name = (item as { name?: unknown }).name
    out.push({
      datasets: stageDatasets,
      epochs,
      ...(typeof name === "string" && name.trim() ? { name } : {}),
    })
  }
  return out
}

// Pick the TrainConfig fields out of a stored blob, defaulting anything missing —
// so a preset saved before a field existed, or a model config that only carries a
// subset, still loads cleanly. Shared by preset-load and model-load. Never trusts
// the blob's shape: each numeric field is coerced.
export function toTrainConfig(
  blob: Record<string, unknown> | null | undefined,
  baseName: string
): TrainConfig {
  const b = blob ?? {}
  const mode = b.mode === "sft" ? "sft" : "pretrain"
  return {
    model_name:
      typeof b.model_name === "string" && b.model_name.trim()
        ? b.model_name
        : baseName,
    dimensions: num(b.dimensions, DEFAULT_CONFIG.dimensions),
    ctx_window: num(b.ctx_window, DEFAULT_CONFIG.ctx_window),
    heads: num(b.heads, DEFAULT_CONFIG.heads),
    layers: num(b.layers, DEFAULT_CONFIG.layers),
    epochs: num(b.epochs, DEFAULT_CONFIG.epochs),
    lr: num(b.lr, DEFAULT_CONFIG.lr),
    ff_multiplier: num(b.ff_multiplier, DEFAULT_CONFIG.ff_multiplier),
    max_vocab: num(b.max_vocab, DEFAULT_CONFIG.max_vocab),
    datasets: datasets(b.datasets),
    curriculum: curriculum(b.curriculum),
    shuffle:
      typeof b.shuffle === "boolean" ? b.shuffle : DEFAULT_CONFIG.shuffle,
    strip_thinking:
      typeof b.strip_thinking === "boolean"
        ? b.strip_thinking
        : DEFAULT_CONFIG.strip_thinking,
    mode,
    base_model_id:
      typeof b.base_model_id === "string" ? b.base_model_id : undefined,
  }
}

// Fetch an existing model's stored training config and map it into a TrainConfig
// for clone-and-tweak. The model's `parameters` is `{ config, dataset }` (see the
// training worker's registration), so we read `parameters.config`. Falls back to
// the model's name for the new run's name.
export async function loadModelConfig(
  modelId: string,
  modelName: string
): Promise<TrainConfig> {
  const res = await fetch(`${API_URL}/models/${modelId}`, {
    headers: authHeaders(),
  })
  if (!res.ok) throw new Error("Failed to load model config.")
  const raw = (await res.json()) as {
    parameters?: { config?: Record<string, unknown> } | null
  }
  return toTrainConfig(raw.parameters?.config, `${modelName}-copy`)
}
