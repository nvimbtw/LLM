import { apiFetch } from "../api"
import { decodeArchitecture, type ContractName } from "./decoder"
import type {
  ArchitectureContracts,
  ErrorResponse,
} from "./contracts.generated"
export interface ArchitectureRequest {
  token: string
  signal?: AbortSignal
}
/** Preserves the upstream status and bounded diagnostic for rejected authoring operations. */
export class ArchitectureApiError extends Error {
  /** Creates an actionable transport error without exposing arbitrary upstream logs. */
  constructor(
    public readonly status: number,
    public readonly detail: ErrorResponse | null
  ) {
    super(detail?.error.message ?? `Architecture request failed (${status}).`)
  }
}
/** Calls Rust with the captured session, validates the response and preserves structured errors. */
export async function architectureRequest<K extends ContractName>(
  path: string,
  model: K,
  request: ArchitectureRequest,
  init: RequestInit = {}
): Promise<ArchitectureContracts[K]> {
  const response = await apiFetch(path, {
    ...init,
    signal: request.signal,
    headers: { Authorization: `Bearer ${request.token}` },
  })
  let value: unknown
  try {
    value = await response.json()
  } catch (error) {
    if (!response.ok) throw new ArchitectureApiError(response.status, null)
    throw error
  }
  if (!response.ok) {
    let detail: ErrorResponse | null = null
    try {
      detail = decodeArchitecture("ErrorResponse", value)
    } catch {
      detail = null
    }
    throw new ArchitectureApiError(response.status, detail)
  }
  return decodeArchitecture(model, value)
}

/** Sends an explicit validation lifecycle command, ignoring its success body while preserving bounded service refusals and captured identity. */
export async function validationCommand(
  path: string,
  method: "POST" | "DELETE",
  request: ArchitectureRequest
): Promise<void> {
  const response = await apiFetch(path, {
    method,
    signal: request.signal,
    headers: { Authorization: `Bearer ${request.token}` },
  })
  if (response.ok) {
    await response.body?.cancel()
    return
  }
  let detail: ErrorResponse | null = null
  try {
    detail = decodeArchitecture("ErrorResponse", await response.json())
  } catch {
    detail = null
  }
  throw new ArchitectureApiError(response.status, detail)
}
/** Serializes validated filters, omitting absent optional values. */
export function architectureQuery(
  model: "CatalogQuery" | "GraphQuery",
  value: unknown
): string {
  const query = decodeArchitecture(model, value)
  return new URLSearchParams(
    Object.entries(query)
      .filter(([, item]) => item != null)
      .map(([key, item]) => [key, String(item)])
  ).toString()
}
