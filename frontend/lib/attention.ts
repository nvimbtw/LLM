import { apiFetch } from "@/lib/api"
import { decodeArchitecture } from "@/lib/architecture/decoder"
import type { ArchitectureRequest } from "@/lib/architecture/http"
import type {
  AttentionSelection,
  InspectionCatalog,
} from "@/lib/architecture/contracts.generated"

export type { InspectionCatalog }

export interface AttentionMatrix {
  modelName: string
  layer: number | null
  head: number
  layers: number | null
  heads: number
  maxTokens: number
  tokens: string[]
  size: number
  weights: Float32Array
  pointId: string | null
  definitionId: string | null
  profileDigest: string | null
  runtimeId: string | null
  evidenceHash: string | null
  validatedSequenceMax: number
}

export interface AttentionCell {
  query: number
  key: number
  weight: number
}

export type AttentionQuery = AttentionSelection

/** Decodes bounded fp32 little-endian observations and rejects nonfinite or invalid probabilities before drawing. */
export function decodeAttention(value: unknown): AttentionMatrix {
  const raw = decodeArchitecture("AttentionMatrix", value)
  const binary = atob(raw.weights)
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0))
  const view = new DataView(bytes.buffer)
  const weights = new Float32Array(bytes.byteLength / 4)
  for (let index = 0; index < weights.length; index++) {
    weights[index] = view.getFloat32(index * 4, true)
  }
  return {
    modelName: raw.model_name,
    layer: raw.layer ?? null,
    head: raw.head,
    layers: raw.layers ?? null,
    heads: raw.heads,
    maxTokens: raw.max_tokens,
    tokens: raw.tokens,
    size: raw.shape[0],
    weights,
    pointId: raw.point_id ?? null,
    definitionId: raw.definition_id ?? null,
    profileDigest: raw.profile_digest ?? null,
    runtimeId: raw.runtime_id ?? null,
    evidenceHash: raw.evidence_hash ?? null,
    validatedSequenceMax: raw.validated_sequence_max ?? 0,
  }
}

/** Extracts a concise refusal from either a direct engine detail or the older nested Rust proxy error. */
function refusal(body: string): string | null {
  const detail = body.match(/"detail"\s*:\s*"((?:[^"\\]|\\.)*)"/)
  if (detail) return JSON.parse(`"${detail[1]}"`) as string
  try {
    const parsed: unknown = JSON.parse(body)
    if (
      parsed &&
      typeof parsed === "object" &&
      "error" in parsed &&
      typeof parsed.error === "string"
    )
      return parsed.error
  } catch {
    return null
  }
  return null
}

/** Calls Rust with the captured session and abort signal; bounded shared decoders interpret only successful data. */
async function attentionResponse(
  path: string,
  request: ArchitectureRequest,
  init: RequestInit = {}
): Promise<unknown> {
  const response = await apiFetch(path, {
    ...init,
    signal: request.signal,
    headers: { Authorization: `Bearer ${request.token}` },
  })
  if (!response.ok)
    throw new Error(
      refusal(await response.text()) ??
        `Unable to inspect attention (${response.status}).`
    )
  return response.json()
}

/** Discovers declared candidates on explicit user action; this may load the saved numerical model into its resident worker. */
export async function fetchInspectionPoints(
  modelId: string,
  request: ArchitectureRequest
): Promise<InspectionCatalog> {
  const catalog = decodeArchitecture(
    "InspectionCatalog",
    await attentionResponse(`/models/${modelId}/attention-points`, request)
  )
  if (catalog.model_name !== modelId)
    throw new Error("Attention metadata belongs to a different model.")
  return catalog
}

/** Reads one selected view and refuses results for a different model, point, legacy layer or head. */
export async function fetchAttention(
  modelId: string,
  query: AttentionQuery,
  request: ArchitectureRequest
): Promise<AttentionMatrix> {
  const selection = decodeArchitecture("AttentionSelection", query)
  const matrix = decodeAttention(
    await attentionResponse(`/models/${modelId}/attention`, request, {
      method: "POST",
      body: JSON.stringify(selection),
    })
  )
  if (
    matrix.modelName !== modelId ||
    matrix.head !== selection.head ||
    (selection.point_id
      ? matrix.pointId !== selection.point_id
      : matrix.layer !== (selection.layer ?? 0))
  )
    throw new Error("Attention result differs from the requested selection.")
  return matrix
}

/** Updates only the measured point's real proof and clears proofs from a retired worker before presenting current metadata. */
export function catalogAfterMeasurement(
  catalog: InspectionCatalog,
  matrix: AttentionMatrix
): InspectionCatalog {
  if (catalog.model_name !== matrix.modelName)
    throw new Error("Attention result belongs to a different model.")
  if (!matrix.pointId) {
    if (
      catalog.legacy_layers !== matrix.layers ||
      catalog.legacy_heads !== matrix.heads
    )
      throw new Error(
        "Attention result differs from the model's legacy dimensions."
      )
    return catalog
  }
  const point = catalog.points?.find((item) => item.id === matrix.pointId)
  if (
    !point ||
    catalog.definition_id !== matrix.definitionId ||
    catalog.profile_digest !== matrix.profileDigest ||
    catalog.max_tokens !== matrix.maxTokens ||
    (point.heads != null && point.heads !== matrix.heads)
  )
    throw new Error("Attention result differs from its declared model view.")
  return {
    ...catalog,
    runtime_id: matrix.runtimeId,
    points: catalog.points!.map((item) =>
      item.id === matrix.pointId
        ? {
            ...item,
            heads: matrix.heads,
            validated_sequence_max: matrix.validatedSequenceMax,
            evidence_hash: matrix.evidenceHash,
          }
        : catalog.runtime_id === matrix.runtimeId
          ? item
          : {
              ...item,
              validated_sequence_max: 0,
              evidence_hash: null,
            }
    ),
  }
}
