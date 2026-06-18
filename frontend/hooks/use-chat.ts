"use client"

import * as React from "react"

import { API_URL, authHeaders } from "@/lib/api"

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
}

export function useChat(chatId: string) {
  const [messages, setMessages] = React.useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [isSending, setIsSending] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    let active = true
    setIsLoading(true)

    async function load() {
      try {
        const response = await fetch(`${API_URL}/chat/${chatId}`, {
          headers: authHeaders(),
        })
        if (!response.ok) throw new Error("Unable to load this chat.")
        const data: ChatMessage[] = await response.json()
        if (active) setMessages(data)
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : "Unable to load this chat.")
        }
      } finally {
        if (active) setIsLoading(false)
      }
    }

    load()
    return () => {
      active = false
    }
  }, [chatId])

  const sendMessage = React.useCallback(
    async (content: string) => {
      setError(null)
      setIsSending(true)
      setMessages((prev) => [...prev, { role: "user", content }])

      try {
        const response = await fetch(`${API_URL}/chat`, {
          method: "POST",
          headers: authHeaders(),
          body: JSON.stringify({ content, chat_id: chatId }),
        })
        if (!response.ok) {
          const text = await response.text()
          throw new Error(text || "Unable to send your message.")
        }
        const data = await response.json()
        setMessages((prev) => [...prev, data.message as ChatMessage])
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to send your message.")
      } finally {
        setIsSending(false)
      }
    },
    [chatId]
  )

  return { messages, isLoading, isSending, error, sendMessage }
}
