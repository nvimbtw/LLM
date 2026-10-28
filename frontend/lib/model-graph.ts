import {
  architectureQuery,
  architectureRequest,
  type ArchitectureRequest,
} from "./architecture/http"
import { decodeArchitecture } from "./architecture/decoder"
import type {
  GraphCursor,
  GraphQuery,
} from "./architecture/contracts.generated"
/** Reads one bounded immutable graph scope and rejects a response for another requested profile or page. */
export async function getModelGraph(
  kind: "models" | "validations",
  id: string,
  query: GraphQuery,
  request: ArchitectureRequest
) {
  const expected = decodeArchitecture("GraphQuery", query)
  const fragment = await architectureRequest(
    `/${kind}/${encodeURIComponent(id)}/graph?${architectureQuery("GraphQuery", expected)}`,
    "GraphFragment",
    request
  )
  if (
    Object.keys(expected).some(
      (field) =>
        expected[field as keyof GraphQuery] !==
        fragment.query[field as keyof GraphQuery]
    )
  )
    throw new Error("Graph response belongs to another scope or profile.")
  return fragment
}
/** Rejects graph cursor replay across artifacts, scopes, profiles or presentation modes. */
export function requireGraphCursor(
  cursor: GraphCursor,
  query: GraphQuery,
  graphHash: string
): void {
  if (
    cursor.graph_hash !== graphHash ||
    ["identity_hash", "view", "mode", "signature", "scope"].some(
      (field) =>
        cursor[field as keyof GraphCursor] !== query[field as keyof GraphQuery]
    )
  )
    throw new Error("Graph cursor belongs to another scope or artifact.")
}
/** Reads immutable resolved definition metadata without importing source or loading weights. */
export function getModelDefinition(id: string, request: ArchitectureRequest) {
  return architectureRequest(
    `/models/${encodeURIComponent(id)}/definition`,
    "ModelDefinitionSummary",
    request
  )
}

/** Reads selected frozen text through Rust and rejects cross-revision/model responses before presentation. */
export async function getRetainedSource(
  kind: "models" | "validations",
  id: string,
  revision: string,
  request: ArchitectureRequest,
  definitionId: string | null = null,
  contentHash: string | null = null
) {
  const value = await architectureRequest(
    `/${kind}/${encodeURIComponent(id)}/sources/${encodeURIComponent(revision)}`,
    "RetainedSource",
    request
  )
  if (
    value.revision_id !== revision ||
    value.definition_id !== definitionId ||
    (contentHash !== null && value.content_hash !== contentHash)
  )
    throw new Error(
      "Source response belongs to another frozen revision or definition."
    )
  return value
}
