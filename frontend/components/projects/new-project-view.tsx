"use client"

import { useRouter } from "next/navigation"
import { ProjectEditor } from "./project-editor"

/** Creates a project from page-native fields and replaces the draft route with its persisted workspace. */
export function NewProjectView() {
  const router = useRouter()
  return (
    <ProjectEditor
      onSaved={(projectId) => router.replace(`/projects/${projectId}`)}
    />
  )
}
