"use client"

import * as React from "react"
import { useSessionToken } from "@/hooks/use-auth"
import { getToken } from "@/lib/api"
import { ProjectCatalog } from "@/lib/project-catalog"
import {
  createProject,
  updateProject,
  deleteProject,
  type ProjectFields,
} from "@/lib/projects"
import { useChats } from "@/components/chat/chats-provider"

interface ProjectsContextValue {
  catalog: ReturnType<ProjectCatalog["getSnapshot"]>
  refresh: () => Promise<void>
  loadMore: () => Promise<void>
  create: (fields: ProjectFields) => Promise<string>
  save: (projectId: string, fields: ProjectFields) => Promise<void>
  remove: (projectId: string) => Promise<void>
  moveChat: (chatId: string, projectId: string | null) => Promise<void>
}

const ProjectsContext = React.createContext<ProjectsContextValue | null>(null)
const EMPTY_CATALOG = {
  token: null,
  projects: [],
  total: 0,
  nextOffset: 0,
  loading: false,
  error: null,
  revision: 0,
}

/** Owns private project reads and coordinates project mutations with the existing chat archive. */
export function ProjectsProvider({ children }: { children: React.ReactNode }) {
  const token = useSessionToken()
  const { refresh: refreshChats, assignProject } = useChats()
  const [owner] = React.useState(() => new ProjectCatalog())
  const snapshot = React.useSyncExternalStore(
    owner.subscribe,
    owner.getSnapshot,
    owner.getSnapshot
  )
  React.useEffect(() => {
    owner.setSession(token)
    return () => owner.dispose()
  }, [owner, token])

  /** Requires the rendered session to remain current before any project mutation. */
  const currentSession = React.useCallback((): string => {
    if (!token || getToken() !== token)
      throw new Error("Your account changed. Please try again.")
    return token
  }, [token])

  /** Refreshes both project membership and ordinary chats after a committed project write. */
  const reconcile = React.useCallback(
    async (owningToken: string): Promise<void> => {
      if (getToken() !== owningToken)
        throw new Error("Your account changed. Please try again.")
      await Promise.all([owner.refresh(), refreshChats()])
    },
    [owner, refreshChats]
  )

  /** Creates a project, reconciles navigation lists and returns its persisted identity. */
  const create = React.useCallback(
    async (fields: ProjectFields): Promise<string> => {
      const owningToken = currentSession()
      const project = await createProject(owningToken, fields)
      await reconcile(owningToken)
      return project.project_id
    },
    [currentSession, reconcile]
  )

  /** Saves editable project metadata and refreshes mounted views after persistence succeeds. */
  const save = React.useCallback(
    async (projectId: string, fields: ProjectFields): Promise<void> => {
      const owningToken = currentSession()
      await updateProject(owningToken, projectId, fields)
      await reconcile(owningToken)
    },
    [currentSession, reconcile]
  )

  /** Deletes a project while retaining its conversations, then reconciles both navigation lists. */
  const remove = React.useCallback(
    async (projectId: string): Promise<void> => {
      const owningToken = currentSession()
      await deleteProject(owningToken, projectId)
      await reconcile(owningToken)
    },
    [currentSession, reconcile]
  )

  /** Moves or removes an existing conversation and refreshes project thread counts and the chat archive. */
  const moveChat = React.useCallback(
    async (chatId: string, projectId: string | null): Promise<void> => {
      const owningToken = currentSession()
      await assignProject(chatId, projectId)
      await reconcile(owningToken)
    },
    [assignProject, currentSession, reconcile]
  )

  const catalog =
    snapshot.token === token
      ? snapshot
      : { ...EMPTY_CATALOG, token, loading: Boolean(token) }
  return (
    <ProjectsContext.Provider
      value={{
        catalog,
        refresh: owner.refresh,
        loadMore: owner.loadMore,
        create,
        save,
        remove,
        moveChat,
      }}
    >
      {children}
    </ProjectsContext.Provider>
  )
}

/** Returns this session's project catalogue and coordinated mutation actions; requires its provider. */
export function useProjects(): ProjectsContextValue {
  const value = React.useContext(ProjectsContext)
  if (!value)
    throw new Error("useProjects must be used within a ProjectsProvider")
  return value
}
