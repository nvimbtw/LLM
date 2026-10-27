import { apiFetch, getToken } from "./api"
import { parseResourceId } from "./chats"
import type { DatasetSelection } from "./training"

export const TOKENIZER_NAME_LIMIT = 120
export const TOKENIZER_DESCRIPTION_LIMIT = 2000

export interface TokenizerDataset {
  id: string
  name: string
  count: number | null
  documents: number
  updated_at: string
}

export interface TokenizerInfo {
  id: string
  name: string
  description: string
  implicit: boolean
  vocabSize: number | null
  merges: number | null
  maxVocab: number | null
  datasets: TokenizerDataset[]
  files: number | null
  bytes: number
  builtAt: string | null
  createdAt: string | null
  fingerprint: string | null
  hasVocab: boolean
}

export interface TokenizerDetail extends TokenizerInfo {
  models: { id: string; name: string }[]
}

export interface TokenizerFields {
  name: string
  description: string
}

export interface TokenizerBuild {
  tokenizer_id: string
  max_vocab: number
  datasets: DatasetSelection[]
}

/** Narrows an interpreted response object, rejecting malformed resource containers. */
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("The server returned invalid vocabulary metadata.")
  return value as Record<string, unknown>
}

/** Reads a required text field without coercing unknown external values. */
function string(value: unknown): string {
  if (typeof value !== "string")
    throw new Error("The server returned invalid vocabulary text.")
  return value
}

/** Validates measured nonnegative counts, preserving null for unavailable measurements. */
function count(value: unknown): number | null {
  if (value === null || value === undefined) return null
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0)
    throw new Error("The server returned an invalid vocabulary count.")
  return value
}

/** Validates and adapts a vocabulary response, keeping optional legacy metadata unavailable. */
export function parseTokenizer(value: unknown): TokenizerInfo {
  const raw = record(value)
  if (
    typeof raw.implicit !== "boolean" ||
    typeof raw.has_vocab !== "boolean" ||
    !Array.isArray(raw.datasets)
  )
    throw new Error("The server returned invalid vocabulary metadata.")
  return {
    id: parseResourceId(raw.tokenizer_id),
    name: string(raw.name),
    description: raw.description == null ? "" : string(raw.description),
    implicit: raw.implicit,
    vocabSize: count(raw.vocab_size),
    merges: count(raw.merges),
    maxVocab: count(raw.max_vocab),
    datasets: raw.datasets.map((value) => {
      const dataset = record(value)
      return {
        id: string(dataset.id),
        name: string(dataset.name),
        count: count(dataset.count),
        documents: count(dataset.documents) ?? 0,
        updated_at: string(dataset.updated_at),
      }
    }),
    files: count(raw.files),
    bytes: count(raw.bytes) ?? 0,
    builtAt: raw.built_at == null ? null : string(raw.built_at),
    createdAt: raw.created_at == null ? null : string(raw.created_at),
    fingerprint: raw.fingerprint == null ? null : string(raw.fingerprint),
    hasVocab: raw.has_vocab,
  }
}

/** Validates detail metadata and its recorded model identities before rendering links. */
export function parseTokenizerDetail(value: unknown): TokenizerDetail {
  const raw = record(value)
  if (!Array.isArray(raw.models))
    throw new Error("The server returned invalid vocabulary model references.")
  return {
    ...parseTokenizer(value),
    models: raw.models.map((value) => {
      const model = record(value)
      return { id: parseResourceId(model.model_id), name: string(model.name) }
    }),
  }
}

/** Trims display fields and validates Unicode character budgets before a metadata write. */
export function validateTokenizerFields(
  fields: TokenizerFields
): TokenizerFields {
  const name = fields.name.trim()
  if (!name || Array.from(name).length > TOKENIZER_NAME_LIMIT)
    throw new Error(`Name must contain 1–${TOKENIZER_NAME_LIMIT} characters.`)
  if (Array.from(fields.description).length > TOKENIZER_DESCRIPTION_LIMIT)
    throw new Error(
      `Description must contain at most ${TOKENIZER_DESCRIPTION_LIMIT} characters.`
    )
  return { name, description: fields.description }
}

/** Reads engine JSON, surfacing actionable service validation and conflict messages. */
async function responseBody(response: Response): Promise<unknown> {
  if (response.status === 409)
    throw new Error(
      "A vocabulary with that name already exists. Choose another name."
    )
  if (response.status === 404)
    throw new Error("This vocabulary no longer exists.")
  const body: unknown = await response.json()
  if (!response.ok) {
    const raw = record(body)
    throw new Error(
      typeof raw.detail === "string"
        ? raw.detail
        : typeof raw.error === "string"
          ? raw.error
          : "Unable to update vocabulary metadata."
    )
  }
  return body
}

/** Fetches vocabulary metadata with an optional captured session; failed or malformed reads reject. */
export async function listTokenizers(
  signal?: AbortSignal,
  token?: string
): Promise<TokenizerInfo[]> {
  const raw = record(
    await responseBody(
      await apiFetch("/tokenizers", {
        signal,
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      })
    )
  )
  if (!Array.isArray(raw.tokenizers))
    throw new Error("The vocabulary catalogue is invalid.")
  return raw.tokenizers.map(parseTokenizer)
}

/** Fetches one vocabulary and recorded model references using its stable identity. */
export async function getTokenizer(
  id: string,
  signal?: AbortSignal
): Promise<TokenizerDetail> {
  return parseTokenizerDetail(
    await responseBody(
      await apiFetch(`/tokenizers/${parseResourceId(id)}`, { signal })
    )
  )
}

/** Saves display metadata for the originating session; rejected writes leave the caller's draft intact. */
export async function editTokenizer(
  token: string,
  id: string,
  fields: TokenizerFields
): Promise<TokenizerDetail> {
  if (getToken() !== token)
    throw new Error("Your account changed. Please try again.")
  const next = validateTokenizerFields(fields)
  const response = await apiFetch(`/tokenizers/${parseResourceId(id)}`, {
    method: "PATCH",
    headers: { Authorization: `Bearer ${token}` },
    body: JSON.stringify({
      display_name: next.name,
      description: next.description,
    }),
  })
  return parseTokenizerDetail(await responseBody(response))
}

/** Deletes a vocabulary and its packed corpora, returning the number of removed packs. */
export async function deleteTokenizer(id: string): Promise<number> {
  const body = record(
    await responseBody(
      await apiFetch(`/tokenizers/${parseResourceId(id)}`, { method: "DELETE" })
    )
  )
  return count(body.corpora_deleted) ?? 0
}

/** Describes the recorded datasets and optional document limits for one vocabulary. */
export function describeDatasets(tokenizer: TokenizerInfo): string {
  if (tokenizer.datasets.length === 0) return "—"
  return tokenizer.datasets
    .map(({ name, count }) => (count ? `${name} (${count})` : name))
    .join(", ")
}
