"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"
import { streamChat } from "@/lib/chat-socket"
import { useChats } from "@/components/chat/chats-provider"
import { useSettings } from "@/components/settings/settings-provider"

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
  // Counted client-side as tokens stream in (see sendMessage below); absent
  // on messages loaded from history, since the backend doesn't persist counts.
  tokenCount?: number
  thinkingTokenCount?: number
  // The model that wrote this reply, for the byline beneath it. Only ever present
  // on assistant messages that came back from the server — a reply still streaming
  // has no byline, and gains one when `onDone` refetches the persisted message.
  modelName?: string
}

// What GET /chat/:id returns per message. Kept separate from ChatMessage because the
// two differ: the wire is snake_case and knows nothing about token counts, which are
// tallied on this side.
interface MessageWire {
  role: "user" | "assistant"
  content: string
  model_name: string | null
}

export function useChat(chatId: string) {
  const { takePendingMessage } = useChats()
  const { settings } = useSettings()
  // Read the system prompt at send time via a ref, so changing it (or any other
  // setting) doesn't churn sendMessage's identity and re-fire the send effects.
  const systemRef = React.useRef(settings.systemPrompt)
  React.useEffect(() => {
    systemRef.current = settings.systemPrompt
  }, [settings.systemPrompt])
  const [messages, setMessages] = React.useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [isSending, setIsSending] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  const fetchMessages = React.useCallback(async () => {
    const response = await apiFetch(`/chat/${chatId}`)
    if (!response.ok) throw new Error("Unable to load this chat.")
    const rows = (await response.json()) as MessageWire[]
    return rows.map<ChatMessage>((row) => ({
      role: row.role,
      content: row.content,
      // Normalized to undefined so the byline's presence check is a single
      // truthiness test rather than having to know about null as well.
      modelName: row.model_name ?? undefined,
    }))
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

      // Counts every token frame the socket delivers for this turn. A token is
      // counted as "thinking" when it falls inside a <|thinking|>…<|/thinking|>
      // block (the opening and closing markers included). We track the block
      // state from the raw token values rather than inferring it from "no
      // visible answer yet" — the model can think after it starts answering,
      // and can open several thinking blocks per reply, so the position of the
      // first visible word tells us nothing reliable about thinking length.
      let totalTokens = 0
      let thinkingTokens = 0
      let insideThinking = false

      const appendToAssistant = (value: string) =>
        setMessages((prev) => {
          const next = [...prev]
          const last = next[next.length - 1]
          if (last?.role === "assistant") {
            totalTokens += 1
            if (value.includes("<|thinking|>")) insideThinking = true
            if (insideThinking) thinkingTokens += 1
            if (value.includes("<|/thinking|>")) insideThinking = false
            // Tokens are whitespace-split words; rejoin with spaces to
            // approximate the structured markup while it streams.
            const sep = last.content ? " " : ""
            next[next.length - 1] = {
              ...last,
              content: last.content + sep + value,
              tokenCount: totalTokens,
              thinkingTokenCount: thinkingTokens,
            }
          }
          return next
        })

      return new Promise<void>((resolve) => {
        streamChat(chatId, content, modelId, systemRef.current, {
          onToken: appendToAssistant,
          onDone: async () => {
            // Replace the streamed approximation with the authoritative,
            // intact content saved on the server, but keep the token counts
            // we tallied — the server doesn't send them back.
            try {
              const fresh = await fetchMessages()
              const lastIdx = fresh.length - 1
              if (fresh[lastIdx]?.role === "assistant") {
                fresh[lastIdx] = {
                  ...fresh[lastIdx],
                  tokenCount: totalTokens,
                  thinkingTokenCount: thinkingTokens,
                }
              }
              setMessages(fresh)
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
