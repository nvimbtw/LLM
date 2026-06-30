"use client"

import * as React from "react"

import { API_URL, authHeaders } from "@/lib/api"
import { streamChat } from "@/lib/chat-socket"

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
}

export function useChat(chatId: string) {
  const [messages, setMessages] = React.useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [isSending, setIsSending] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  const fetchMessages = React.useCallback(async () => {
    const response = await fetch(`${API_URL}/chat/${chatId}`, {
      headers: authHeaders(),
    })
    if (!response.ok) throw new Error("Unable to load this chat.")
    return (await response.json()) as ChatMessage[]
  }, [chatId])

  React.useEffect(() => {
    let active = true
    setIsLoading(true)
    fetchMessages()
      .then((data) => {
        if (active) setMessages(data)
      })
      .catch((err) => {
        if (active)
          setError(err instanceof Error ? err.message : "Unable to load this chat.")
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
    }
  }, [fetchMessages])

  const sendMessage = React.useCallback(
    (content: string, modelId: string) => {
      setError(null)
      setIsSending(true)
      // User message + an empty assistant message we fill as tokens arrive.
      setMessages((prev) => [
        ...prev,
        { role: "user", content },
        { role: "assistant", content: "" },
      ])

      const appendToAssistant = (value: string) =>
        setMessages((prev) => {
          const next = [...prev]
          const last = next[next.length - 1]
          if (last?.role === "assistant") {
            // Tokens are whitespace-split words; rejoin with spaces to
            // approximate the structured markup while it streams.
            const sep = last.content ? " " : ""
            next[next.length - 1] = {
              ...last,
              content: last.content + sep + value,
            }
          }
          return next
        })

      return new Promise<void>((resolve) => {
        streamChat(chatId, content, modelId, {
          onToken: appendToAssistant,
          onDone: async () => {
            // Replace the streamed approximation with the authoritative,
            // intact content saved on the server.
            try {
              setMessages(await fetchMessages())
            } catch {
              // Keep the approximation if the refetch fails.
            }
            setIsSending(false)
            resolve()
          },
          onError: (message) => {
            setError(message)
            setIsSending(false)
            resolve()
          },
        })
      })
    },
    [chatId, fetchMessages]
  )

  return { messages, isLoading, isSending, error, sendMessage }
}
