"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { getToken } from "@/lib/api"
import { useProjects } from "@/components/projects/projects-provider"
import {
  validateProjectFields,
  type Project,
  type ProjectFields,
} from "@/lib/projects"

/** Owns a project form's editable fields and persistence; stale account completions cannot close or navigate its replacement. */
export function useProjectEditor(
  project: Project | undefined,
  onSaved: (projectId: string) => void
) {
  const token = useSessionToken()
  const { create, save } = useProjects()
  const [fields, setFields] = React.useState<ProjectFields>(() => ({
    name: project?.name ?? "",
    description: project?.description ?? "",
    notes: project?.notes ?? "",
  }))
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
  const change = React.useCallback(
    (field: keyof ProjectFields, value: string): void => {
      setFields((current) => ({ ...current, [field]: value }))
    },
    []
  )

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
      if (active.current && getToken() === token) onSaved(projectId)
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
  return { fields, change, submitting, error, submit }
}
