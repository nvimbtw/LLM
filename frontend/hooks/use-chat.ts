"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"
import { streamChat } from "@/lib/chat-socket"
import { setMessageFeedback, type Feedback } from "@/lib/feedback"
import { useChats } from "@/components/chat/chats-provider"
import { useSettings } from "@/components/settings/settings-provider"
import type { ThinkBudget } from "@/components/chat/budget-selector"

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
  // Server id. Absent on a reply that is still streaming and on the optimistic
  // user message, which is exactly why rating is gated on it: a reply with no id
  // yet has no row to attach a verdict to.
  id?: string
  // The thumbs the user has given this reply, or undefined for none.
  feedback?: Feedback
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
  id_message: string
  role: "user" | "assistant"
  content: string
  model_name: string | null
  feedback: Feedback | null
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
  // How many jobs sit in front of this turn, once it has been queued. Null when
  // nothing is in flight; 0 means it is next (or already generating). A reply is a
  // job on the GPU queue now, so a turn really can wait — and saying so is the
  // difference between "waiting its turn" and "broken".
  const [queuedAhead, setQueuedAhead] = React.useState<number | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  // Stops the in-flight turn. Held in a ref rather than state because nothing
  // renders from it — it only needs to survive between the send and the stop.
  const cancelRef = React.useRef<(() => void) | null>(null)

  const fetchMessages = React.useCallback(async () => {
    const response = await apiFetch(`/chat/${chatId}`)
    if (!response.ok) throw new Error("Unable to load this chat.")
    const rows = (await response.json()) as MessageWire[]
    return rows.map<ChatMessage>((row) => ({
      id: row.id_message,
      role: row.role,
      content: row.content,
      // Normalized to undefined so the byline's presence check is a single
      // truthiness test rather than having to know about null as well.
      modelName: row.model_name ?? undefined,
      feedback: row.feedback ?? undefined,
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
          setError(
            err instanceof Error ? err.message : "Unable to load this chat."
          )
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
    (content: string, modelId: string, budget: ThinkBudget) => {
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
        cancelRef.current = streamChat(
          chatId,
          content,
          modelId,
          systemRef.current,
          budget,
          {
            onQueued: ({ ahead }) => setQueuedAhead(ahead),
            onToken: (value) => {
              // The first token is proof the turn reached the GPU, whatever the
              // queue said a moment ago.
              setQueuedAhead(0)
              appendToAssistant(value)
            },
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
              setQueuedAhead(null)
              resolve()
            },
            onError: (message) => {
              setError(message)
              setIsSending(false)
              setQueuedAhead(null)
              resolve()
            },
          }
        )
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
    if (pending) sendMessage(pending.content, pending.modelId, pending.budget)
  }, [loadedOnce, chatId, takePendingMessage, sendMessage])

  // Rate a reply, or clear the rating by passing the value it already has —
  // clicking thumbs-up twice means "never mind", not "up, but harder".
  //
  // Applied optimistically and rolled back on failure. A thumbs is a one-click
  // aside from whatever the user was actually doing; making them wait on a round
  // trip to see the button fill in would cost more than the write is worth, and a
  // silent no-op on a dropped request would quietly lose training data.
  const rateMessage = React.useCallback(
    async (messageId: string, feedback: Feedback) => {
      let previous: Feedback | undefined
      let next: Feedback | undefined
      setMessages((prev) =>
        prev.map((message) => {
          if (message.id !== messageId) return message
          previous = message.feedback
          next = message.feedback === feedback ? undefined : feedback
          return { ...message, feedback: next }
        })
      )

      try {
        await setMessageFeedback(messageId, next ?? null)
      } catch {
        setMessages((prev) =>
          prev.map((message) =>
            message.id === messageId
              ? { ...message, feedback: previous }
              : message
          )
        )
        setError("Your feedback could not be saved.")
      }
    },
    []
  )

  const cancel = React.useCallback(() => {
    cancelRef.current?.()
    cancelRef.current = null
  }, [])

  return {
    messages,
    isLoading,
    isSending,
    queuedAhead,
    error,
    sendMessage,
    cancel,
    rateMessage,
  }
}
