import { architectures } from "../architectures"
import { blocks } from "../blocks"
import { draftDigest } from "./draft"
import type { ArchitectureRequest } from "./http"

export interface CopiedDraft {
  name: string
  description: string
  source: string
}

/** Reads one exact immutable custom revision or the eligible wrapper of an installed builtin, without creating or rewriting a resource. */
export async function readCopiedDraft(
  kind: "block" | "architecture",
  id: string,
  revision: string,
  request: ArchitectureRequest
): Promise<CopiedDraft> {
  const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
  if (
    typeof id !== "string" ||
    typeof revision !== "string" ||
    !uuid.test(id) ||
    !uuid.test(revision)
  )
    throw new Error("Copy requires an exact resource and accepted revision.")
  const api = kind === "block" ? blocks : architectures
  const detail = await api.detail(id, request)
  if (detail.summary.resource_id !== id || detail.summary.kind !== kind)
    throw new Error("Copy metadata belongs to another resource.")
  if (kind === "block" && detail.summary.origin === "builtin") {
    const copied = await blocks.copySource(id, revision, request)
    return { ...copied, description: copied.description ?? "" }
  }
  const view = await api.revision(id, revision, request)
  const source = "source" in view ? view.source : view.yaml
  const reference = view.accepted.reference
  if (
    reference.resource_id !== id ||
    reference.revision_id !== revision ||
    reference.content_hash !== draftDigest(source)
  )
    throw new Error(
      "Copied source differs from the selected immutable revision."
    )
  return {
    name: `${detail.summary.name} copy`.slice(0, 120),
    description: detail.summary.description,
    source,
  }
}
