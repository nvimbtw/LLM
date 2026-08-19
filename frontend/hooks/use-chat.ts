"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"
import { streamChat, type SamplingSettings } from "@/lib/chat-socket"
import { parseMessage, textContent, toolOf } from "@/lib/message-parser"
import type { ToolAnswer } from "@/components/chat/trace"
import { setMessageFeedback, type Feedback } from "@/lib/feedback"
import { useChats } from "@/components/chat/chats-provider"
import { useSettings } from "@/components/settings/settings-provider"
import type { ThinkBudget } from "@/components/chat/budget-selector"

export interface ChatMessage {
  role: "user" | "assistant"
  /** The answer. For a turn that used a tool, the *last* round's reply. */
  content: string
  /**
   * The working that preceded the answer — the first round's reply, holding its
   * reasoning and the calls it made. Absent on a turn that never called
   * anything, whose reasoning is inside `content` as it always was.
   */
  working?: string
  /** What each call in `working` came back with, in the order they closed. */
  toolAnswers?: ToolAnswer[]
  /** Seconds the whole turn took, summed across its rounds. */
  elapsedSecs?: number
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
  message_id: string
  // `tool` is a stored row like any other — see `groupTurns`, which folds it
  // back into the assistant turn it belongs to.
  role: "user" | "assistant" | "tool"
  content: string
  model_name: string | null
  feedback: Feedback | null
}

/**
 * The persisted rows of one turn, folded into the single message it was.
 *
 * A reply that calls a tool is stored as `assistant → tool → assistant`: the
 * round that asked, what came back, and the round that answered. That is the
 * right shape for the *model*, whose next prompt has to contain all three, and
 * the wrong shape for a reader, who asked one question and got one answer. So
 * the last assistant round is the message and everything before it is its
 * working.
 *
 * Rows arrive in creation order, which is the order this depends on.
 */
function groupTurns(rows: MessageWire[]): ChatMessage[] {
  const grouped: ChatMessage[] = []

  for (const row of rows) {
    const previous = grouped[grouped.length - 1]

    if (row.role === "tool") {
      // A tool row with no assistant before it cannot be placed. It should not
      // exist; dropping it is better than inventing a turn to hang it on.
      if (previous?.role !== "assistant") continue
      previous.working ??= previous.content
      previous.toolAnswers = [
        ...(previous.toolAnswers ?? []),
        parseToolRow(row.content),
      ]
      previous.content = ""
      continue
    }

    // The round after a tool result continues the turn rather than starting one.
    if (
      row.role === "assistant" &&
      previous?.toolAnswers &&
      !previous.content
    ) {
      previous.content = row.content
      previous.id = row.message_id
      previous.modelName = row.model_name ?? undefined
      previous.feedback = row.feedback ?? undefined
      continue
    }

    grouped.push({
      id: row.message_id,
      role: row.role,
      content: row.content,
      // Normalized to undefined so the byline's presence check is a single
      // truthiness test rather than having to know about null as well.
      modelName: row.model_name ?? undefined,
      feedback: row.feedback ?? undefined,
    })
  }

  return grouped
}

/**
 * A stored tool row back into the call it answered and the answer.
 *
 * The row opens by repeating the call — `<|web-search|>query<|/web-search|>`
 * then the result — which is what tells the model which of its calls it is
 * being handed (see `engine/llm/chat/tools.py`). Reading it back here is the
 * same fact used the other way round.
 */
function parseToolRow(content: string): ToolAnswer {
  for (const node of parseMessage(content).children) {
    const tool = toolOf(node)
    if (!tool) continue
    const at = content.indexOf(`<|/${tool}|>`)
    return {
      tool,
      argument: textContent(node).trim(),
      body: at === -1 ? content : content.slice(at + `<|/${tool}|>`.length),
    }
  }
  // A row that names no call is still something the model was told.
  return { tool: "", argument: "", body: content }
}

export function useChat(chatId: string) {
  const { takePendingMessage, refresh: refreshChats } = useChats()
  const { settings } = useSettings()
  // Read the system prompt at send time via a ref, so changing it (or any other
  // setting) doesn't churn sendMessage's identity and re-fire the send effects.
  const systemRef = React.useRef(settings.systemPrompt)
  React.useEffect(() => {
    systemRef.current = settings.systemPrompt
  }, [settings.systemPrompt])
  // The decoding knobs go through a ref for the same reason, and are read at send
  // time so a mid-conversation change applies to the next turn without the
  // in-flight one being re-sent.
  const samplingRef = React.useRef<SamplingSettings>({
    temperature: settings.temperature,
    topK: settings.topK,
    topP: settings.topP,
  })
  React.useEffect(() => {
    samplingRef.current = {
      temperature: settings.temperature,
      topK: settings.topK,
      topP: settings.topP,
    }
  }, [settings.temperature, settings.topK, settings.topP])
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
    const response = await apiFetch(`/chats/${chatId}`)
    if (!response.ok) throw new Error("Unable to load this chat.")
    const { messages: rows } = (await response.json()) as {
      messages: MessageWire[]
    }
    return groupTurns(rows)
  }, [chatId])

  // Flips true once the initial DB load settles. The pending-first-message send
  // waits for this so its optimistic messages aren't clobbered by the load.
  const [loadedOnce, setLoadedOnce] = React.useState(false)

  // Navigating between two chats reuses this hook, so the load flags have to go
  // back to their initial values for the new id. Done during render, not in the
  // effect: `loadedOnce` gates the pending-first-message send, and a frame where
  // it is still true for the *previous* chat is a frame in which that send can
  // fire against the wrong conversation.
  const [loadedFor, setLoadedFor] = React.useState(chatId)
  if (loadedFor !== chatId) {
    setLoadedFor(chatId)
    setIsLoading(true)
    setLoadedOnce(false)
  }

  React.useEffect(() => {
    let active = true
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

      /**
       * The first call of the turn retires what has streamed so far into the
       * trace, and everything after it is the answer.
       *
       * Done on the *call* rather than on the round boundary because the call is
       * the first moment we know this turn has a working phase at all. Until
       * then a reply is just a reply, and the common case never grows a trace.
       */
      const beginWorking = () =>
        setMessages((prev) => {
          const next = [...prev]
          const last = next[next.length - 1]
          if (last?.role === "assistant" && last.working === undefined) {
            next[next.length - 1] = {
              ...last,
              working: last.content,
              content: "",
            }
          }
          return next
        })

      const recordAnswer = (answer: ToolAnswer) =>
        setMessages((prev) => {
          const next = [...prev]
          const last = next[next.length - 1]
          if (last?.role === "assistant") {
            next[next.length - 1] = {
              ...last,
              toolAnswers: [...(last.toolAnswers ?? []), answer],
            }
          }
          return next
        })

      /**
       * Tokens arriving after the first call belong to the answer; the ones
       * before it have already been moved into `working`, and the markup of the
       * call itself has to keep landing there so the trace can see it.
       */
      const appendToWorking = (value: string) =>
        setMessages((prev) => {
          const next = [...prev]
          const last = next[next.length - 1]
          if (last?.role === "assistant" && last.working !== undefined) {
            totalTokens += 1
            const sep = last.working ? " " : ""
            next[next.length - 1] = {
              ...last,
              working: last.working + sep + value,
              tokenCount: totalTokens,
            }
          }
          return next
        })

      // Which half of the turn tokens are currently filling. `working` latches
      // on the first call; `answering` flips once the first result lands, which
      // is when the round that called is over.
      let working = false
      let answering = false
      let elapsedSecs = 0

      return new Promise<void>((resolve) => {
        cancelRef.current = streamChat(
          chatId,
          content,
          modelId,
          systemRef.current,
          budget,
          samplingRef.current,
          {
            onToken: (value: string) => {
              setQueuedAhead(0)
              if (working && !answering) appendToWorking(value)
              else appendToAssistant(value)
            },
            onToolCall: () => {
              // The markup spelling the call has already streamed into the
              // reply, so there is nothing to add — this only marks where the
              // working phase begins.
              working = true
              beginWorking()
            },
            onToolResult: (result) => {
              answering = true
              recordAnswer(result)
            },
            onDone: async (info) => {
              // A turn's elapsed time is the sum of the rounds it took, not the
              // last one's — the whole of it is what the reader waited.
              elapsedSecs += info.elapsedSecs ?? 0
              // One `done` per round. Only the last is the turn, and refetching
              // on an earlier one would replace a reply still being written
              // with the half of it the engine has stored so far.
              if (!info.final) return
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
                    // Also not persisted: the stored rows carry per-round
                    // usage, but nothing records what the turn as a whole took.
                    elapsedSecs: elapsedSecs || undefined,
                  }
                }
                setMessages(fresh)
              } catch {
                // Keep the approximation if the refetch fails.
              }
              // The sidebar's per-chat token total (and the composer status
              // row's fallback to it once streaming stops, see chat-thread.tsx)
              // both read the conversations list, which otherwise only
              // refreshes on create/rename/delete. Without this, a finished
              // turn's tokens never land there — the sidebar count goes
              // stale, and the composer's "used" figure visibly drops back to
              // that stale count the moment `isSending` flips false, since
              // the live tally it was showing during streaming is gone but
              // the persisted total hasn't caught up yet. Awaited so both
              // land in the same render as `isSending` going false, rather
              // than one frame after — that's the difference between a
              // number that settles and one that visibly dips first.
              await refreshChats()
              setIsSending(false)
              setQueuedAhead(null)
              resolve()
            },
            onError: (message: string) => {
              setError(message)
              setIsSending(false)
              setQueuedAhead(null)
              resolve()
            },
          }
        )
      })
    },
    [chatId, fetchMessages, refreshChats]
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
