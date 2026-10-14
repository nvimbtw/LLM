"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { getToken } from "@/lib/api"
import {
  loadProject,
  projectThreadOffset,
  type ProjectDetail,
} from "@/lib/projects"
import { useProjects } from "@/components/projects/projects-provider"

interface ProjectViewState {
  token: string
  projectId: string
  offset: number
  project: ProjectDetail | null
  error: string | null
}

/** Loads one account/project/page identity, cancelling navigation reads and hiding stale private data immediately. */
export function useProject(projectId: string) {
  const token = useSessionToken()
  const { catalog } = useProjects()
  const [page, setPage] = React.useState({ projectId, offset: 0 })
  const offset = page.projectId === projectId ? page.offset : 0
  const [reload, setReload] = React.useState(0)
  const [state, setState] = React.useState<ProjectViewState | null>(null)
  const [settledRevision, setSettledRevision] = React.useState("")
  const revision = `${catalog.revision}:${reload}`

  React.useEffect(() => {
    if (!token) return
    const controller = new AbortController()
    loadProject(token, projectId, offset, controller.signal)
      .then((project) => {
        if (controller.signal.aborted || getToken() !== token) return
        const validOffset = projectThreadOffset(offset, project.threads_total)
        if (validOffset !== offset) {
          setPage({ projectId, offset: validOffset })
          return
        }
        setState({ token, projectId, offset, project, error: null })
        setSettledRevision(revision)
      })
      .catch((failure: unknown) => {
        if (controller.signal.aborted || getToken() !== token) return
        setState({
          token,
          projectId,
          offset,
          project: null,
          error:
            failure instanceof Error
              ? failure.message
              : "Unable to load this project.",
        })
        setSettledRevision(revision)
      })
    return () => controller.abort()
  }, [token, projectId, offset, revision])

  const current =
    state?.token === token &&
    state.projectId === projectId &&
    state.offset === offset
      ? state
      : null
  return {
    project: current?.project ?? null,
    error: current?.error ?? null,
    isLoading: Boolean(token) && (!current || settledRevision !== revision),
    offset,
    changePage: (nextOffset: number) =>
      setPage({ projectId, offset: nextOffset }),
    refresh: () => setReload((value) => value + 1),
  }
}
