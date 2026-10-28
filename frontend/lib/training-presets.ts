import { apiFetch } from "./api"
import { DEFAULT_CONFIG } from "@/hooks/use-training"
import type { CurriculumStage, DatasetSelection, TrainConfig } from "./training"
import { decodeArchitecture } from "./architecture/decoder"
import { withoutTrainingReceipt } from "./training-selection"

// A named, reusable training-parameter set. `config` is a TrainConfig-shaped blob
// the backend stores opaquely (validated only when a run actually starts). Mirrors
// the response from routes/preset.rs.
export interface Preset {
  id_preset: string
  name: string
  config: Record<string, unknown>
  updated_at: string
}

export interface PresetRequest {
  token: string
  signal?: AbortSignal
}

/** Validates a stored preset envelope before it may replace any local training form. */
function parsePreset(value: unknown): Preset {
  if (!value || typeof value !== "object")
    throw new Error("Preset response is invalid.")
  const row = value as Record<string, unknown>
  if (
    typeof row.id_preset !== "string" ||
    typeof row.name !== "string" ||
    !row.config ||
    typeof row.config !== "object" ||
    Array.isArray(row.config) ||
    typeof row.updated_at !== "string"
  )
    throw new Error("Preset response is invalid.")
  return {
    id_preset: row.id_preset,
    name: row.name,
    config: row.config as Record<string, unknown>,
    updated_at: row.updated_at,
  }
}

/** Retains reusable immutable pins/settings in version two while requiring fresh validation after loading. */
function storedConfiguration(config: TrainConfig): Record<string, unknown> {
  return { schema_version: 2, config: withoutTrainingReceipt(config) }
}

/** Reads this captured session's preset envelopes; malformed/failed reads remain visible errors instead of a fabricated empty list. */
export async function listPresets(request?: PresetRequest): Promise<Preset[]> {
  const res = await apiFetch("/presets", {
    signal: request?.signal,
    headers: request ? { Authorization: `Bearer ${request.token}` } : undefined,
  })
  if (!res.ok) throw new Error("Unable to load training presets.")
  const values: unknown = await res.json()
  if (!Array.isArray(values)) throw new Error("Preset catalogue is invalid.")
  return values.map(parsePreset)
}

/** Creates a named version-two preset for the originating session, preserving exact pins without carrying authorization receipts. */
export async function createPreset(
  name: string,
  config: TrainConfig,
  request?: PresetRequest
): Promise<Preset> {
  const res = await apiFetch("/presets", {
    method: "POST",
    signal: request?.signal,
    headers: request ? { Authorization: `Bearer ${request.token}` } : undefined,
    body: JSON.stringify({ name, config: storedConfiguration(config) }),
  })
  if (!res.ok) {
    throw new Error(
      res.status === 409
        ? "A preset with that name already exists."
        : (await res.text()) || "Failed to save preset."
    )
  }
  return parsePreset(await res.json())
}

/** Saves supplied preset fields using the same versioned envelope and captured identity as creation. */
export async function updatePreset(
  id: string,
  patch: { name?: string; config?: TrainConfig },
  request?: PresetRequest
): Promise<Preset> {
  const res = await apiFetch(`/presets/${encodeURIComponent(id)}`, {
    method: "PUT",
    signal: request?.signal,
    headers: request ? { Authorization: `Bearer ${request.token}` } : undefined,
    body: JSON.stringify({
      ...patch,
      ...(patch.config ? { config: storedConfiguration(patch.config) } : {}),
    }),
  })
  if (!res.ok) {
    throw new Error(
      res.status === 409
        ? "A preset with that name already exists."
        : (await res.text()) || "Failed to update preset."
    )
  }
  return parsePreset(await res.json())
}

/** Deletes only this explicit preset under its originating identity; a missing row is already deleted. */
export async function deletePreset(
  id: string,
  request?: PresetRequest
): Promise<void> {
  const res = await apiFetch(`/presets/${encodeURIComponent(id)}`, {
    method: "DELETE",
    signal: request?.signal,
    headers: request ? { Authorization: `Bearer ${request.token}` } : undefined,
  })
  if (!res.ok && res.status !== 404) {
    throw new Error("Failed to delete preset.")
  }
}

/** Adapts legacy sparse numeric settings using explicit form defaults; selection identities are validated independently. */
function num(v: unknown, fallback: number): number {
  return typeof v === "number" && Number.isFinite(v) ? v : fallback
}

/** Adapts old dataset choices to durable IDs/nonnegative document limits, dropping obsolete name-only selections. */
function datasets(v: unknown): DatasetSelection[] {
  if (!Array.isArray(v)) return []
  const out: DatasetSelection[] = []
  for (const item of v) {
    if (!item || typeof item !== "object") continue
    const id = (item as { id?: unknown }).id
    if (typeof id !== "string" || !id.trim()) continue
    const name = (item as { name?: unknown }).name
    const rawCount = (item as { count?: unknown }).count
    const count =
      typeof rawCount === "number" && Number.isFinite(rawCount) && rawCount > 0
        ? Math.floor(rawCount)
        : 0
    out.push({
      id,
      ...(typeof name === "string" && name.trim() ? { name } : {}),
      count,
    })
  }
  return out
}

/** Adapts sparse legacy curriculum stages while preserving order and dropping stages with no durable dataset selections. */
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

/** Adapts legacy sparse settings or version-two presets, retaining exact typed recipe pins and requiring fresh evidence; malformed/unknown versions reject without replacing the form. */
export function toTrainConfig(
  blob: Record<string, unknown> | null | undefined,
  baseName: string
): TrainConfig {
  let b = blob ?? {}
  if (b.schema_version === 2) {
    if (!b.config || typeof b.config !== "object" || Array.isArray(b.config))
      throw new Error("Version-two preset has invalid training settings.")
    b = b.config as Record<string, unknown>
  } else if (b.schema_version != null && b.schema_version !== 1)
    throw new Error("This training preset version is unsupported.")
  const mode =
    b.mode === "sft" || b.mode === "kto" || b.mode === "raft"
      ? b.mode
      : typeof b.base_model === "string"
        ? "sft"
        : "pretrain"
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
    batch_size: num(b.batch_size, DEFAULT_CONFIG.batch_size),
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
      typeof b.base_model_id === "string"
        ? b.base_model_id
        : typeof b.base_model === "string"
          ? b.base_model
          : undefined,
    resume_from: typeof b.resume_from === "string" ? b.resume_from : undefined,
    tokenizer_id:
      typeof b.tokenizer_id === "string" ? b.tokenizer_id : undefined,
    architecture:
      b.architecture == null
        ? undefined
        : {
            ...decodeArchitecture("ArchitectureSelection", b.architecture),
            validation_run_id: null,
          },
    precision:
      b.precision === "fp32" || b.precision === "fp16" || b.precision === "bf16"
        ? b.precision
        : "bf16",
    seed: num(b.seed, 0),
    allow_tf32: typeof b.allow_tf32 === "boolean" ? b.allow_tf32 : true,
    checkpoint_target: num(
      b.checkpoint_target,
      DEFAULT_CONFIG.checkpoint_target ?? 10
    ),
    beta: num(b.beta, DEFAULT_CONFIG.beta ?? 0.1),
  }
}

/** Loads reusable model provenance without treating missing modular selections as a default standard shape; imported models without settings require explicit continuation from weights. */
export async function loadModelConfig(
  modelId: string,
  modelName: string,
  request?: PresetRequest
): Promise<TrainConfig> {
  const res = await apiFetch(`/models/${encodeURIComponent(modelId)}`, {
    signal: request?.signal,
    headers: request ? { Authorization: `Bearer ${request.token}` } : undefined,
  })
  if (!res.ok) throw new Error("Failed to load model config.")
  const raw = (await res.json()) as {
    parameters?: { config?: Record<string, unknown> } | null
    definition_id?: string | null
  }
  const config = toTrainConfig(raw.parameters?.config, `${modelName}-copy`)
  if (
    raw.definition_id &&
    !config.architecture &&
    !config.base_model_id &&
    !config.resume_from
  )
    throw new Error(
      "This saved model has no reusable training selection. Continue from its weights instead."
    )
  return config
}
