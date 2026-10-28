"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { getToken } from "@/lib/api"
import { usePageDraft } from "./use-page-draft"
import { readDraftStrings } from "@/lib/page-draft"
import { useProjects } from "@/components/projects/projects-provider"
import {
  validateProjectFields,
  type Project,
  type ProjectFields,
} from "@/lib/projects"

/** Restores only the project's three editable text fields, including incomplete or multiline values. */
function projectDraft(value: unknown): ProjectFields {
  return readDraftStrings(["name", "description", "notes"], value)
}

/** Owns a project form's editable fields and persistence; stale account completions cannot close or navigate its replacement. */
export function useProjectEditor(
  project: Project | undefined,
  onSaved: (projectId: string) => void
) {
  const token = useSessionToken()
  const { create, save } = useProjects()
  const draft = usePageDraft(
    `project:${project?.project_id ?? "new"}`,
    {
      name: project?.name ?? "",
      description: project?.description ?? "",
      notes: project?.notes ?? "",
    },
    projectDraft
  )
  const fields = draft.fields
  const [submitting, setSubmitting] = React.useState(false)
  const submittingRef = React.useRef(false)
  const [error, setError] = React.useState<string | null>(null)
  const active = React.useRef(true)
  React.useEffect(() => {
    active.current = true
    return () => {
      active.current = false
    }
  }, [])

  /** Updates one named draft field without writing private metadata before submission. */
  const change = (field: keyof ProjectFields, value: string): void =>
    draft.change({ ...fields, [field]: value })

  /** Discards local changes and restores the current saved project fields. */
  const reset = () => {
    if (submittingRef.current) return
    draft.reset()
    setError(null)
  }

  /** Validates and persists this form, reporting failures beside the controls and preserving the draft for retry. */
  const submit = async (): Promise<void> => {
    if (submittingRef.current || !token) return
    submittingRef.current = true
    setSubmitting(true)
    setError(null)
    try {
      const next = validateProjectFields(fields)
      let projectId: string
      if (project) {
        await save(project.project_id, next)
        projectId = project.project_id
      } else projectId = await create(next)
      if (active.current && getToken() === token) {
        draft.change(next)
        draft.forget()
        onSaved(projectId)
      }
    } catch (failure) {
      if (active.current && getToken() === token)
        setError(
          failure instanceof Error
            ? failure.message
            : "Unable to save this project."
        )
    } finally {
      submittingRef.current = false
      if (active.current && getToken() === token) setSubmitting(false)
    }
  }
  const hasChanges =
    fields.name !== (project?.name ?? "") ||
    fields.description !== (project?.description ?? "") ||
    fields.notes !== (project?.notes ?? "")
  return {
    fields,
    change,
    reset,
    submitting,
    error: error ?? draft.persistenceError,
    submit,
    hasChanges,
  }
}
