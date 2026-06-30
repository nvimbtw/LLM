"use client"

import * as React from "react"

import { API_URL, authHeaders } from "@/lib/api"
import { streamChat } from "@/lib/chat-socket"

export interface ChatSummary {
  id: string
  title: string
  createdAt: string
}

interface ChatsContextValue {
  conversations: ChatSummary[]
  isLoading: boolean
  error: string | null
  refresh: () => Promise<void>
  createChat: (message: string, modelId: string) => Promise<string>
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
      const response = await fetch(`${API_URL}/chat`, { headers: authHeaders() })
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

  React.useEffect(() => {
    refresh()
  }, [refresh])

  const createChat = React.useCallback(
    (message: string, modelId: string) =>
      new Promise<string>((resolve, reject) => {
        // The first message is streamed too, but here we only need the new
        // chat id; the chat page loads the saved messages after navigation.
        streamChat(null, message, modelId, {
          onToken: () => {},
          onDone: async ({ chatId }) => {
            await refresh()
            resolve(chatId)
          },
          onError: (m) => reject(new Error(m)),
        })
      }),
    [refresh]
  )

  const renameChat = React.useCallback(
    async (id: string, title: string) => {
      const response = await fetch(`${API_URL}/chat/${id}`, {
        method: "PUT",
        headers: authHeaders(),
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
      const response = await fetch(`${API_URL}/chat/${id}`, {
        method: "DELETE",
        headers: authHeaders(),
      })
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
