"use client"

import * as React from "react"

import { SESSION_CHANGED_EVENT, apiFetch, getToken } from "@/lib/api"

export interface ChatSummary {
  id: string
  title: string
  createdAt: string
}

export interface PendingMessage {
  content: string
  modelId: string
}

interface ChatsContextValue {
  conversations: ChatSummary[]
  isLoading: boolean
  error: string | null
  refresh: () => Promise<void>
  createChat: (message: string, modelId: string) => Promise<string>
  // The first message of a freshly-created chat, handed off to the chat page so
  // it can stream the reply there (one-shot: returns and clears it).
  takePendingMessage: (chatId: string) => PendingMessage | null
  renameChat: (id: string, title: string) => Promise<void>
  deleteChat: (id: string) => Promise<void>
}

const ChatsContext = React.createContext<ChatsContextValue | null>(null)

interface RawChat {
  id_chat: string | number
  title: string
  created_at: string
}

function mapChat(raw: RawChat): ChatSummary {
  return { id: String(raw.id_chat), title: raw.title, createdAt: raw.created_at }
}

export function ChatsProvider({ children }: { children: React.ReactNode }) {
  const [conversations, setConversations] = React.useState<ChatSummary[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  const refresh = React.useCallback(async () => {
    try {
      const response = await apiFetch(`/chat`)
      if (!response.ok) throw new Error("Unable to load your chats.")
      const data: RawChat[] = await response.json()
      setConversations(data.map(mapChat))
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load your chats.")
    } finally {
      setIsLoading(false)
    }
  }, [])

  // This provider mounts in the root layout — before login — so the mount-time
  // fetch can run unauthenticated and go stale. Refetch when the session
  // changes (login), and drop the list when it's cleared (logout).
  React.useEffect(() => {
    refresh()
    const onSessionChanged = () => {
      if (getToken()) {
        refresh()
      } else {
        setConversations([])
      }
    }
    window.addEventListener(SESSION_CHANGED_EVENT, onSessionChanged)
    return () => window.removeEventListener(SESSION_CHANGED_EVENT, onSessionChanged)
  }, [refresh])

  // Pending first messages, keyed by the new chat id. A ref (not state) so the
  // value is written synchronously before navigation and read once on the chat
  // page's mount, with no re-render churn.
  const pendingRef = React.useRef<Map<string, PendingMessage>>(new Map())

  const takePendingMessage = React.useCallback((chatId: string) => {
    const msg = pendingRef.current.get(chatId)
    if (msg) pendingRef.current.delete(chatId)
    return msg ?? null
  }, [])

  // Create the chat row only (no generation) so we can navigate immediately;
  // the chat page streams the reply for the stashed first message on mount.
  const createChat = React.useCallback(
    async (message: string, modelId: string): Promise<string> => {
      const response = await apiFetch(`/chat`, {
        method: "POST",
        body: JSON.stringify({ content: message }),
      })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || "Unable to start a chat.")
      }
      const id = String((await response.json()).id_chat)
      pendingRef.current.set(id, { content: message, modelId })
      // Surface the new chat in the sidebar; no need to block navigation on it.
      refresh()
      return id
    },
    [refresh]
  )

  const renameChat = React.useCallback(
    async (id: string, title: string) => {
      const response = await apiFetch(`/chat/${id}`, {
        method: "PUT",
        body: JSON.stringify({ title }),
      })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || "Unable to rename the chat.")
      }
      await refresh()
    },
    [refresh]
  )

  const deleteChat = React.useCallback(
    async (id: string) => {
      const response = await apiFetch(`/chat/${id}`, { method: "DELETE" })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || "Unable to delete the chat.")
      }
      await refresh()
    },
    [refresh]
  )

  const value: ChatsContextValue = {
    conversations,
    isLoading,
    error,
    refresh,
    createChat,
    takePendingMessage,
    renameChat,
    deleteChat,
  }

  return <ChatsContext.Provider value={value}>{children}</ChatsContext.Provider>
}

export function useChats() {
  const ctx = React.useContext(ChatsContext)
  if (!ctx) {
    throw new Error("useChats must be used within a ChatsProvider")
  }
  return ctx
}
