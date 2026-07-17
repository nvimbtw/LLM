"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"
import { streamChat } from "@/lib/chat-socket"
import { useChats } from "@/components/chat/chats-provider"

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
}

export function useChat(chatId: string) {
  const { takePendingMessage } = useChats()
  const [messages, setMessages] = React.useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [isSending, setIsSending] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  const fetchMessages = React.useCallback(async () => {
    const response = await apiFetch(`/chat/${chatId}`)
    if (!response.ok) throw new Error("Unable to load this chat.")
    return (await response.json()) as ChatMessage[]
  }, [chatId])

  // Flips true once the initial DB load settles. The pending-first-message send
  // waits for this so its optimistic messages aren't clobbered by the load.
  const [loadedOnce, setLoadedOnce] = React.useState(false)

  React.useEffect(() => {
    let active = true
    setIsLoading(true)
    setLoadedOnce(false)
    fetchMessages()
      .then((data) => {
        if (active) setMessages(data)
      })
      .catch((err) => {
        if (active)
          setError(err instanceof Error ? err.message : "Unable to load this chat.")
      })
      .finally(() => {
        if (active) {
          setIsLoading(false)
          setLoadedOnce(true)
        }
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

  // A chat just created from the home screen hands its first message off here;
  // stream the reply once the (empty) initial load has settled. takePendingMessage
  // is one-shot, so React's double-invoked effects / remounts won't re-send.
  React.useEffect(() => {
    if (!loadedOnce) return
    const pending = takePendingMessage(chatId)
    // Intentional: starting the first turn here is the whole point of the
    // handoff, and takePendingMessage is one-shot so it can't loop.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (pending) sendMessage(pending.content, pending.modelId)
  }, [loadedOnce, chatId, takePendingMessage, sendMessage])

  return { messages, isLoading, isSending, error, sendMessage }
}
