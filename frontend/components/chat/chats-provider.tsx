"use client"

import * as React from "react"

import type { ThinkBudget } from "./budget-selector"

import { apiFetch, getToken } from "@/lib/api"
import { useAuth } from "@/hooks/use-auth"

export interface ChatSummary {
  id: string
  title: string
  createdAt: string
  /** Cumulative tokens across the chat's messages (0 for a brand-new chat). */
  tokens: number
}

export interface PendingMessage {
  content: string
  modelId: string
  // Carried across the navigation into the new chat, so the budget picked in the
  // composer is the one the first turn is actually generated under.
  budget: ThinkBudget
}

interface ChatsContextValue {
  conversations: ChatSummary[]
  isLoading: boolean
  error: string | null
  refresh: () => Promise<void>
  createChat: (
    message: string,
    modelId: string,
    budget: ThinkBudget
  ) => Promise<string>
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
  tokens?: number
}

function mapChat(raw: RawChat): ChatSummary {
  return {
    id: String(raw.id_chat),
    title: raw.title,
    createdAt: raw.created_at,
    tokens: raw.tokens ?? 0,
  }
}

const NO_CHATS: ChatSummary[] = []

export function ChatsProvider({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth()
  const [conversations, setConversations] = React.useState<ChatSummary[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  const refresh = React.useCallback(async () => {
    // This provider lives in the root layout, so it mounts on unauthenticated
    // pages (/login, /signup) too. Skip the fetch when there's no token: an
    // unauthenticated /chat call 401s, which would redirect to /login and,
    // once the token is cleared, spin in an endless reload loop.
    if (!getToken()) return
    try {
      const response = await apiFetch(`/chat`)
      if (!response.ok) throw new Error("Unable to load your chats.")
      const data: RawChat[] = await response.json()
      setConversations(data.map(mapChat))
      setError(null)
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unable to load your chats."
      )
    } finally {
      setIsLoading(false)
    }
  }, [])

  // The session drives the load. `useAuth` reads it as an external store now, so
  // logging in re-runs this effect and logging out re-renders with an empty list
  // *derived* below — which is why this hook no longer keeps its own
  // `SESSION_CHANGED_EVENT` listener, and no longer has to remember to clear
  // state on the way out.
  //
  // The fetch is inlined rather than calling `refresh`, so no state is written
  // synchronously from the effect body; `refresh` stays for the mutations, which
  // only run authenticated. Same shape as `use-signals.ts`.
  React.useEffect(() => {
    if (!isAuthenticated) return
    let active = true
    apiFetch(`/chat`)
      .then(async (response) => {
        if (!response.ok) throw new Error("Unable to load your chats.")
        return (await response.json()) as RawChat[]
      })
      .then((data) => {
        if (!active) return
        setConversations(data.map(mapChat))
        setError(null)
      })
      .catch((err: unknown) => {
        if (!active) return
        setError(
          err instanceof Error ? err.message : "Unable to load your chats."
        )
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
    }
  }, [isAuthenticated])

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
    async (
      message: string,
      modelId: string,
      budget: ThinkBudget
    ): Promise<string> => {
      const response = await apiFetch(`/chat`, {
        method: "POST",
        body: JSON.stringify({ content: message }),
      })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || "Unable to start a chat.")
      }
      const id = String((await response.json()).id_chat)
      pendingRef.current.set(id, { content: message, modelId, budget })
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
    // Logged out is derived, not stored: the previous session's titles are still
    // sitting in `conversations` until the next load replaces them, and a
    // sidebar that keeps showing them after a logout is the bug this prevents.
    conversations: isAuthenticated ? conversations : NO_CHATS,
    isLoading: isAuthenticated && isLoading,
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
