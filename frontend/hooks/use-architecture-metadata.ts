"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { usePageDraft } from "./use-page-draft"
import { readDraftStrings } from "@/lib/page-draft"
import type { ArchitectureKind } from "./use-architecture-catalog"
import { architectures } from "@/lib/architectures"
import { blocks } from "@/lib/blocks"
import { encodeArchitecture } from "@/lib/architecture/decoder"
import type {
  ResourceDetail,
  ResourceSummary,
} from "@/lib/architecture/contracts.generated"

/** Restores independently owned display/archive metadata without treating source bytes as metadata. */
function metadataDraft(value: unknown) {
  const text = readDraftStrings(["name", "description"], value)
  if (
    !value ||
    typeof value !== "object" ||
    !("archived" in value) ||
    typeof value.archived !== "boolean"
  )
    throw new Error("The saved archive choice is invalid.")
  return { ...text, archived: value.archived }
}

/** Edits one optimistic metadata version without changing or replacing the source document owned by its workspace. */
export function useArchitectureMetadata(
  kind: ArchitectureKind,
  summary: ResourceSummary,
  onSaved: (detail: ResourceDetail) => void
) {
  const token = useSessionToken()
  const draft = usePageDraft(
    `metadata:${kind}:${summary.resource_id}`,
    {
      name: summary.name,
      description: summary.description,
      archived: summary.archived,
    },
    metadataDraft
  )
  const { name, description, archived } = draft.fields
  const [busy, setBusy] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const active = React.useRef(false)
  const pending = React.useRef<AbortController | null>(null)
  React.useEffect(() => {
    active.current = true
    return () => {
      active.current = false
      pending.current?.abort()
    }
  }, [])

  /** Updates only name/description/archive with the captured metadata version; concurrent edits remain a visible refusal. */
  async function save() {
    if (!token || pending.current || summary.origin === "builtin") return
    const value = {
      expected_metadata_version: summary.metadata_version,
      name,
      description,
      archived,
    }
    try {
      encodeArchitecture("MetadataPatch", value)
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Check the metadata fields."
      )
      return
    }
    const controller = new AbortController()
    pending.current = controller
    setBusy(true)
    setError(null)
    try {
      const result = await (kind === "block" ? blocks : architectures).patch(
        summary.resource_id,
        value,
        { token, signal: controller.signal }
      )
      if (
        result.summary.resource_id !== summary.resource_id ||
        result.summary.kind !== kind ||
        result.summary.metadata_version !== summary.metadata_version + 1 ||
        result.summary.name !== name.trim() ||
        result.summary.description !== description ||
        result.summary.archived !== archived
      )
        throw new Error(
          "Metadata response differs from the submitted resource update."
        )
      if (active.current) {
        draft.forget()
        onSaved(result)
      }
    } catch (error) {
      if (active.current)
        setError(
          error instanceof Error ? error.message : "Metadata did not save."
        )
    } finally {
      if (pending.current === controller) pending.current = null
      if (active.current) setBusy(false)
    }
  }

  return {
    name,
    setName: (name: string) => draft.change({ ...draft.fields, name }),
    description,
    setDescription: (description: string) =>
      draft.change({ ...draft.fields, description }),
    archived,
    setArchived: (archived: boolean) =>
      draft.change({ ...draft.fields, archived }),
    busy,
    error: error ?? draft.persistenceError,
    save,
    reset: () => {
      if (!pending.current) {
        draft.reset()
        setError(null)
      }
    },
    dirty:
      name !== summary.name ||
      description !== summary.description ||
      archived !== summary.archived,
  }
}
