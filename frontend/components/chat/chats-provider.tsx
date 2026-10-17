"use client"

import * as React from "react"
import type { ThinkBudget } from "@/lib/think-budget"
import { apiFetch, getToken } from "@/lib/api"
import type { Attachment } from "@/lib/attachments"
import { useSessionToken } from "@/hooks/use-auth"
import { invalidateChatHistory } from "@/lib/chat-history"
import {
  parseChatSummary,
  parseResourceId,
  assignChatProject,
  type ChatSummary,
} from "@/lib/chats"
import { ChatCatalog } from "@/lib/chat-catalog"
import { PendingChatMessages, type PendingMessage } from "@/lib/pending-chat"

export type { ChatSummary } from "@/lib/chats"
export type { PendingMessage } from "@/lib/pending-chat"

interface ChatsContextValue {
  conversations: ChatSummary[]
  isLoading: boolean
  error: string | null
  archive: ChatSummary[]
  isArchiveLoading: boolean
  archiveError: string | null
  loadArchive: () => Promise<void>
  catalog: ChatCatalog
  refresh: () => Promise<void>
  createChat: (
    message: string,
    modelId: string,
    budget: ThinkBudget,
    attachments: Attachment[],
    projectId?: string
  ) => Promise<string>
  takePendingMessage: (chatId: string) => PendingMessage | null
  renameChat: (id: string, title: string) => Promise<void>
  deleteChat: (id: string) => Promise<void>
  assignProject: (id: string, projectId: string | null) => Promise<void>
}

const ChatsContext = React.createContext<ChatsContextValue | null>(null)
const NO_CHATS: ChatSummary[] = []
const TITLE_MAX = 60

/** Derives the sidebar title from the opening message's first line, bounded to 60 characters. */
function openingTitle(message: string): string {
  const line = message.trim().split("\n")[0] || "New chat"
  return line.length > TITLE_MAX ? `${line.slice(0, TITLE_MAX - 1)}…` : line
}

/** Owns recent chats, on-demand archive reads and pending first-message handoff for one account. */
export function ChatsProvider({ children }: { children: React.ReactNode }) {
  const token = useSessionToken()
  const [catalog] = React.useState(() => new ChatCatalog())
  const [pending] = React.useState(() => new PendingChatMessages())
  const snapshot = React.useSyncExternalStore(
    catalog.subscribe,
    catalog.getSnapshot,
    catalog.getSnapshot
  )
  const refresh = catalog.refresh
  React.useEffect(() => {
    pending.setSession(token)
    catalog.setSession(token)
    return () => catalog.dispose()
  }, [token, pending, catalog])

  /** Requires a current account before writes and routes may use their results. */
  const requireSession = React.useCallback((): string => {
    if (!token || getToken() !== token)
      throw new Error("Your account changed. Please try again.")
    return token
  }, [token])

  /** Takes one session-owned first message after navigation to its conversation. */
  const takePendingMessage = React.useCallback(
    (chatId: string): PendingMessage | null => pending.take(getToken(), chatId),
    [pending]
  )

  /** Creates an ordinary or project conversation row and carries the real composer's first message into its streaming page. */
  const createChat = React.useCallback(
    async (
      message: string,
      modelId: string,
      budget: ThinkBudget,
      attachments: Attachment[] = [],
      projectId?: string
    ): Promise<string> => {
      const owningToken = requireSession()
      const response = await apiFetch("/chats", {
        method: "POST",
        headers: { Authorization: `Bearer ${owningToken}` },
        body: JSON.stringify({
          title: openingTitle(message),
          ...(projectId ? { project_id: parseResourceId(projectId) } : {}),
        }),
      })
      if (!response.ok) throw new Error("Unable to start a chat.")
      const chat = parseChatSummary(await response.json())
      if (getToken() !== owningToken)
        throw new Error("Your account changed. Please try again.")
      pending.setSession(owningToken)
      pending.put(owningToken, chat.id, {
        content: message,
        modelId,
        budget,
        attachments,
      })
      catalog.updateSummary(chat)
      void refresh()
      return chat.id
    },
    [catalog, pending, refresh, requireSession]
  )

  /** Renames a stored conversation, then refreshes archive titles for the owning account. */
  const renameChat = React.useCallback(
    async (id: string, title: string): Promise<void> => {
      const owningToken = requireSession()
      const response = await apiFetch(`/chats/${id}`, {
        method: "PATCH",
        headers: { Authorization: `Bearer ${owningToken}` },
        body: JSON.stringify({ title }),
      })
      if (!response.ok) throw new Error("Unable to rename the chat.")
      const chat = parseChatSummary(await response.json())
      if (getToken() === owningToken) {
        invalidateChatHistory(id)
        catalog.updateSummary(chat)
        await refresh()
      }
    },
    [catalog, refresh, requireSession]
  )

  /** Deletes a stored conversation and invalidates its prefetched history before refreshing the archive. */
  const deleteChat = React.useCallback(
    async (id: string): Promise<void> => {
      const owningToken = requireSession()
      const response = await apiFetch(`/chats/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${owningToken}` },
      })
      if (!response.ok) throw new Error("Unable to delete the chat.")
      invalidateChatHistory(id)
      if (getToken() === owningToken) {
        catalog.remove(id)
        await refresh()
      }
    },
    [catalog, refresh, requireSession]
  )

  /** Changes only project membership; messages and title remain stored with the same conversation identity. */
  const assignProject = React.useCallback(
    async (id: string, projectId: string | null): Promise<void> => {
      const owningToken = requireSession()
      const chat = await assignChatProject(owningToken, id, projectId)
      if (getToken() !== owningToken)
        throw new Error("Your account changed. Please try again.")
      invalidateChatHistory(id)
      catalog.updateSummary(chat)
    },
    [catalog, requireSession]
  )

  const current = snapshot.token === token ? snapshot : null
  return (
    <ChatsContext.Provider
      value={{
        conversations: current?.recent ?? NO_CHATS,
        isLoading: Boolean(token) && (!current || !current.recentLoaded),
        error: current?.recentError ?? null,
        archive: current?.archive ?? NO_CHATS,
        isArchiveLoading:
          Boolean(token) &&
          (!current || (current.archive === null && !current.archiveError)),
        archiveError: current?.archiveError ?? null,
        loadArchive: catalog.loadArchive,
        catalog,
        refresh,
        createChat,
        takePendingMessage,
        renameChat,
        deleteChat,
        assignProject,
      }}
    >
      {children}
    </ChatsContext.Provider>
  )
}

/** Returns the session's archive and mutation actions; requires a ChatsProvider. */
export function useChats(): ChatsContextValue {
  const value = React.useContext(ChatsContext)
  if (!value) throw new Error("useChats must be used within a ChatsProvider")
  return value
}

/** Retains the complete archive only while its page or picker is open, including account replacements. */
export function useChatArchive(open = true) {
  const value = useChats()
  const token = useSessionToken()
  React.useEffect(() => {
    if (open) return value.catalog.retainArchive()
  }, [open, token, value.catalog])
  return {
    conversations: value.archive,
    isLoading: value.isArchiveLoading,
    error: value.archiveError,
    refresh: value.loadArchive,
  }
}
