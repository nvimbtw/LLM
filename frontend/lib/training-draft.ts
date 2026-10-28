import { apiFetch, getToken } from "./api"
import { decodeArchitecture, identityDigest } from "./architecture/decoder"
import { withoutTrainingReceipt } from "./training-selection"
import type { TrainConfig, TrainingStep } from "./training"

export const TRAINING_DRAFT_CHANGED = "training-draft-changed"
const MAX_DRAFT_CHARACTERS = 262144
const STORAGE_UNAVAILABLE = new Error(
  "Browser storage is unavailable. Your training draft cannot be saved."
)

export interface TrainingDraft {
  version: 1
  config: TrainConfig
  steps: TrainingStep[]
}

export interface TrainingDraftIdentity {
  token: string
  accountId: string
}

/** Builds a browser storage key from the authenticated profile's durable UUID, never from the opaque bearer token. */
export function trainingDraftKey(accountId: string): string {
  if (
    !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(
      accountId
    )
  )
    throw new Error("The current account has no browser draft identity.")
  return `slate-training-draft:${accountId}`
}

/** Resolves an opaque captured session through the existing profile endpoint; failed or malformed identity reads never authorize draft access. */
export async function loadTrainingDraftAccount(
  token: string,
  signal?: AbortSignal
): Promise<string> {
  const response = await apiFetch("/user/me", {
    headers: { Authorization: `Bearer ${token}` },
    signal,
  })
  if (!response.ok)
    throw new Error("Your account could not be loaded for training drafts.")
  const value: unknown = await response.json()
  if (!record(value) || typeof value.id_user !== "string")
    throw new Error("Your account's draft identity is invalid.")
  trainingDraftKey(value.id_user)
  return value.id_user
}

/** Narrows interpreted local JSON to a non-array object, rejecting invalid containers. */
function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

/** Restores null numeric placeholders as invalid/empty form values, preserving incomplete edits for subsequent validation. */
function draftNumber(value: unknown): number {
  if (value === null) return Number.NaN
  if (typeof value !== "number" || !Number.isFinite(value))
    throw new Error("The saved training draft contains an invalid number.")
  return value
}

/** Decodes dataset picks without normalizing invalid limits or discarding empty curriculum stages. */
function draftDatasets(value: unknown): TrainConfig["datasets"] {
  if (!Array.isArray(value))
    throw new Error("The saved training draft contains invalid datasets.")
  return value.map((item) => {
    if (
      !record(item) ||
      typeof item.id !== "string" ||
      (item.name !== undefined && typeof item.name !== "string")
    )
      throw new Error("The saved training draft contains an invalid dataset.")
    return { id: item.id, name: item.name, count: draftNumber(item.count) }
  })
}

/** Validates the local form shape without applying submission constraints or replacing unfinished values with defaults. Restored pins require fresh engine validation. */
function draftConfig(value: unknown): TrainConfig {
  if (
    !record(value) ||
    typeof value.model_name !== "string" ||
    typeof value.shuffle !== "boolean" ||
    typeof value.strip_thinking !== "boolean" ||
    (value.mode !== "pretrain" &&
      value.mode !== "sft" &&
      value.mode !== "kto" &&
      value.mode !== "raft") ||
    !Array.isArray(value.curriculum)
  )
    throw new Error("The saved training draft contains invalid settings.")
  const config: TrainConfig = {
    model_name: value.model_name,
    mode: value.mode,
    shuffle: value.shuffle,
    strip_thinking: value.strip_thinking,
    dimensions: draftNumber(value.dimensions),
    ctx_window: draftNumber(value.ctx_window),
    heads: draftNumber(value.heads),
    layers: draftNumber(value.layers),
    epochs: draftNumber(value.epochs),
    lr: draftNumber(value.lr),
    batch_size: draftNumber(value.batch_size),
    ff_multiplier: draftNumber(value.ff_multiplier),
    max_vocab: draftNumber(value.max_vocab),
    datasets: draftDatasets(value.datasets),
    curriculum: value.curriculum.map((stage) => {
      if (
        !record(stage) ||
        (stage.name !== undefined && typeof stage.name !== "string")
      )
        throw new Error(
          "The saved training draft contains an invalid curriculum stage."
        )
      return {
        name: stage.name,
        epochs: draftNumber(stage.epochs),
        datasets: draftDatasets(stage.datasets),
      }
    }),
  }
  for (const key of ["seed", "beta", "checkpoint_target"] as const)
    if (value[key] !== undefined) config[key] = draftNumber(value[key])
  for (const key of ["base_model_id", "resume_from", "tokenizer_id"] as const) {
    if (value[key] === undefined) continue
    if (typeof value[key] !== "string")
      throw new Error(
        "The saved training draft contains an invalid resource selection."
      )
    config[key] = value[key]
  }
  if (value.precision !== undefined) {
    if (
      value.precision !== "fp32" &&
      value.precision !== "fp16" &&
      value.precision !== "bf16"
    )
      throw new Error("The saved training draft contains an invalid precision.")
    config.precision = value.precision
  }
  if (value.allow_tf32 !== undefined) {
    if (typeof value.allow_tf32 !== "boolean")
      throw new Error(
        "The saved training draft contains an invalid TF32 setting."
      )
    config.allow_tf32 = value.allow_tf32
  }
  return withoutTrainingReceipt({
    ...config,
    architecture:
      value.architecture === undefined
        ? undefined
        : decodeArchitecture("ArchitectureSelection", value.architecture),
  })
}

/** Decodes a bounded, versioned browser draft. Corruption rejects visibly rather than silently replacing the form or stored bytes. */
export function parseTrainingDraft(raw: string): TrainingDraft {
  if (raw.length > MAX_DRAFT_CHARACTERS)
    throw new Error("The saved training draft is too large.")
  const value: unknown = JSON.parse(raw)
  if (!record(value) || value.version !== 1 || !Array.isArray(value.steps))
    throw new Error(
      "This saved training draft is invalid or uses an unsupported version."
    )
  const ids = new Set<string>()
  const steps = value.steps.map((step): TrainingStep => {
    if (
      !record(step) ||
      typeof step.id !== "string" ||
      ids.has(step.id) ||
      (step.kind !== "train" && step.kind !== "benchmark") ||
      (step.suite !== "language" &&
        step.suite !== "language_v3" &&
        step.suite !== "markup" &&
        step.suite !== "assistant")
    )
      throw new Error("The saved training draft contains an invalid follow-up.")
    ids.add(step.id)
    return {
      id: step.id,
      kind: step.kind,
      suite: step.suite,
      train: draftConfig(step.train),
      errors: [],
    }
  })
  return { version: 1, config: draftConfig(value.config), steps }
}

/** Encodes exact reusable form settings and follow-up order, excluding validation receipts/errors; nonfinite numeric edits remain null placeholders. */
export function serializeTrainingDraft(
  config: TrainConfig,
  steps: TrainingStep[]
): string {
  const raw = JSON.stringify({
    version: 1,
    config: withoutTrainingReceipt(config),
    steps: steps.map((step) => ({
      id: step.id,
      kind: step.kind,
      suite: step.suite,
      train: withoutTrainingReceipt(step.train),
    })),
  })
  if (raw.length > MAX_DRAFT_CHARACTERS)
    throw new Error("The training draft is too large to save in this browser.")
  return raw
}

/** Returns a stable raw snapshot for React; inaccessible storage produces a stable error instead of falsely reporting an empty draft. */
export function readTrainingDraft(
  identity: TrainingDraftIdentity | null
): string | null | Error {
  if (!identity) return null
  try {
    return localStorage.getItem(trainingDraftKey(identity.accountId))
  } catch {
    return STORAGE_UNAVAILABLE
  }
}

/** Saves only for the still-current bearer, then notifies this tab. Storage failures leave the editor's unsaved state available to retry. */
export function writeTrainingDraft(
  identity: TrainingDraftIdentity,
  config: TrainConfig,
  steps: TrainingStep[]
): void {
  if (getToken() !== identity.token)
    throw new Error("Your session changed. This draft was not saved.")
  localStorage.setItem(
    trainingDraftKey(identity.accountId),
    serializeTrainingDraft(config, steps)
  )
  window.dispatchEvent(new Event(TRAINING_DRAFT_CHANGED))
}

/** Compares complete snapshots independent of JSON object key order; invalid data is removable only by exact raw equality. */
function sameDraft(
  actual: string | null | Error,
  expected: string | null | Error
): boolean {
  if (actual === expected) return true
  if (typeof actual !== "string" || typeof expected !== "string") return false
  if (
    actual.length > MAX_DRAFT_CHARACTERS ||
    expected.length > MAX_DRAFT_CHARACTERS
  )
    return false
  try {
    return (
      identityDigest(JSON.parse(actual)) ===
      identityDigest(JSON.parse(expected))
    )
  } catch {
    return false
  }
}

/** Removes only the snapshot acknowledged by this session, preserving edits saved by another tab while a submission was in flight. */
export function removeTrainingDraft(
  identity: TrainingDraftIdentity,
  expected: string | null | Error
): boolean {
  if (
    getToken() !== identity.token ||
    !sameDraft(readTrainingDraft(identity), expected)
  )
    return false
  localStorage.removeItem(trainingDraftKey(identity.accountId))
  window.dispatchEvent(new Event(TRAINING_DRAFT_CHANGED))
  return true
}
