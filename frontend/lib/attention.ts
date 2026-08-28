import { apiFetch } from "@/lib/api"

/**
 * One head's attention over one piece of text.
 *
 * `weights` is row-major and `size × size`: row `i` is what the token at
 * position `i` attended to, so everything above the diagonal is zero — a token
 * cannot attend to its own future. `tokens` labels both axes, because they are
 * the same axis.
 */
export interface AttentionMatrix {
  layer: number
  head: number
  layers: number
  heads: number
  /** The engine's ceiling for this model, in tokens. */
  maxTokens: number
  tokens: string[]
  size: number
  weights: Float32Array
}

export interface AttentionCell {
  query: number
  key: number
  weight: number
}

export interface AttentionQuery {
  text: string
  layer: number
  head: number
}

interface RawAttention {
  layer: number
  head: number
  layers: number
  heads: number
  max_tokens: number
  tokens: string[]
  shape: [number, number]
  dtype: string
  encoding: string
  weights: string
}

const BYTES_PER_FLOAT = 4

// The engine sends the matrix as base64 fp32 rather than JSON numbers: at the
// 512-token ceiling that is 1.33 MiB on the wire against 2.5–3 MiB for the same
// values printed to six decimals, and it needs no decision about how much
// precision to discard to get there. See engine/llm/orchestrate/attention.py.
const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1

function decodeFloat32(base64: string): Float32Array {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)
  // The engine writes little-endian; a typed array reads host order. Every
  // platform a browser runs on today is little-endian, so the second branch will
  // never execute — but a matrix read byte-swapped is not a failure anyone would
  // recognise as one, and the guard is four lines.
  if (LITTLE_ENDIAN) return new Float32Array(bytes.buffer)
  const view = new DataView(bytes.buffer)
  const values = new Float32Array(bytes.byteLength / BYTES_PER_FLOAT)
  for (let i = 0; i < values.length; i += 1) {
    values[i] = view.getFloat32(i * BYTES_PER_FLOAT, true)
  }
  return values
}

/**
 * The sentence worth showing a person, dug out of a proxied refusal.
 *
 * `upstream.rs` reports an engine 400 as `{"error": "service returned 400:
 * {\"detail\": …}"}` — every proxy route in the backend answers that way, so
 * unwrapping it is this side's job rather than a shape to change there.
 */
function refusal(body: string): string | null {
  const detail = body.match(/"detail"\s*:\s*"((?:[^"\\]|\\.)*)"/)
  if (detail) return JSON.parse(`"${detail[1]}"`) as string
  try {
    const parsed: unknown = JSON.parse(body)
    if (parsed && typeof parsed === "object" && "error" in parsed) {
      return String((parsed as { error: unknown }).error)
    }
  } catch {
    // A body that is not JSON tells us nothing the status has not.
  }
  return null
}

export async function fetchAttention(
  modelId: string,
  query: AttentionQuery
): Promise<AttentionMatrix> {
  const response = await apiFetch(`/models/${modelId}/attention`, {
    method: "POST",
    body: JSON.stringify(query),
  })
  if (!response.ok) {
    throw new Error(
      refusal(await response.text()) ?? "Unable to read this model's attention."
    )
  }

  const raw = (await response.json()) as RawAttention
  const size = raw.shape[0]
  const weights = decodeFloat32(raw.weights)
  if (weights.length !== size * size) {
    throw new Error("The attention matrix arrived the wrong size.")
  }

  return {
    layer: raw.layer,
    head: raw.head,
    layers: raw.layers,
    heads: raw.heads,
    maxTokens: raw.max_tokens,
    tokens: raw.tokens,
    size,
    weights,
  }
}
