"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"
import { streamChat, type SamplingSettings } from "@/lib/chat-socket"
import {
  fetchContextFloor,
  getContextFloor,
  setContextFloor,
  type StoredFloor,
} from "@/lib/context-floor"
import {
  mapAttachment,
  type Attachment,
  type AttachmentWire,
} from "@/lib/attachments"
import { callOf, parseMessage, TOOL_CALL_CLOSE } from "@/lib/message-parser"
import type { ToolAnswer } from "@/components/chat/trace"
import { setMessageFeedback, type Feedback } from "@/lib/feedback"
import { useChats } from "@/components/chat/chats-provider"
import { useComposerChoice } from "@/components/chat/composer-choice-provider"
import { useSettings } from "@/components/settings/settings-provider"
import type { ThinkBudget } from "@/lib/think-budget"

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
  // Files sent with the question. Only ever on a user turn: the model is not
  // given them and so cannot answer with one.
  attachments?: Attachment[]
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
  attachments: AttachmentWire[]
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
/**
 * One round's markup added to the working so far.
 *
 * Separated by a space rather than concatenated flush: the two are independent
 * pieces of markup, and a round ending in `</|par|>` against one opening in
 * `<|thinking|>` parses either way, but a round ending mid-sentence against the
 * next one's first word does not read as two.
 */
function joinWorking(working: string | undefined, round: string): string {
  if (working === undefined) return round
  if (!round) return working
  return working ? `${working} ${round}` : round
}

function groupTurns(rows: MessageWire[]): ChatMessage[] {
  const grouped: ChatMessage[] = []

  for (const row of rows) {
    const previous = grouped[grouped.length - 1]

    if (row.role === "tool") {
      // A tool row with no assistant before it cannot be placed. It should not
      // exist; dropping it is better than inventing a turn to hang it on.
      if (previous?.role !== "assistant") continue
      // Appended, not assigned once. A turn can take several rounds, and every
      // round before the one that finally answers is working — so the second
      // tool row has a round of its own to fold in, not just the first. `??=`
      // here dropped it: `working` was already set, and clearing `content`
      // below then lost that round's reasoning and its call outright.
      previous.working = joinWorking(previous.working, previous.content)
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
      // Absent rather than empty, so nothing downstream has to distinguish
      // "no files" from "a row written before attachments existed".
      attachments: row.attachments?.length
        ? row.attachments.map(mapAttachment)
        : undefined,
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
 * The row opens by repeating the call —
 * `<|tool-call|>{"tool": …, "argument": …}<|/tool-call|>` then the result —
 * which is what tells the model which of its calls it is being handed (see
 * `engine/llm/chat/tools.py`). Reading it back here is the same fact used the
 * other way round.
 */
function parseToolRow(content: string): ToolAnswer {
  for (const node of parseMessage(content).children) {
    const call = callOf(node)
    if (!call) continue
    const at = content.indexOf(TOOL_CALL_CLOSE)
    return {
      ...call,
      body: at === -1 ? content : content.slice(at + TOOL_CALL_CLOSE.length),
    }
  }
  // A row that names no call is still something the model was told.
  return { tool: "", argument: "", body: content }
}

export function useChat(chatId: string) {
  const { takePendingMessage, refresh: refreshChats } = useChats()
  const { settings } = useSettings()
  // The model actually answering right now — a stored floor sized to a
  // different model's `ctx_window` doesn't apply to this one. See the note
  // on `StoredFloor` and where `historyFloor` is computed in `sendMessage`.
  const { model: currentModel } = useComposerChoice()
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
    contextMultiplier: settings.contextMultiplier,
  })
  React.useEffect(() => {
    samplingRef.current = {
      temperature: settings.temperature,
      topK: settings.topK,
      topP: settings.topP,
      contextMultiplier: settings.contextMultiplier,
    }
  }, [settings.temperature, settings.topK, settings.topP, settings.contextMultiplier])

  // What `/prune` last found for this chat — browser-local, not the engine's
  // to remember (see `lib/context-floor.ts`). Re-read whenever the chat
  // changes, and held in state (not just the ref below) so the footer's token
  // count re-renders the moment a prune actually lands.
  const [floor, setFloor] = React.useState<StoredFloor | null>(() => getContextFloor(chatId))
  // Adjusted during render rather than in an Effect, same as the composer
  // footer's own title-follows-prop pattern — see composer-footer.tsx.
  const [floorChatId, setFloorChatId] = React.useState(chatId)
  if (chatId !== floorChatId) {
    setFloorChatId(chatId)
    setFloor(getContextFloor(chatId))
  }
  // The ref `sendMessage` actually reads — a plain `useCallback` closing over
  // `floor` would go stale for the lifetime of one render, and this is read on
  // every turn.
  const floorRef = React.useRef<StoredFloor | null>(null)
  React.useEffect(() => {
    floorRef.current = floor
  }, [floor])

  const pruneContext = React.useCallback(
    async (modelId: string) => {
      const found = await fetchContextFloor(chatId, modelId)
      if (found.prunedTurns > 0 && found.floorMessageId) {
        const stored = {
          messageId: found.floorMessageId,
          prunedTokens: found.prunedTokens,
          modelId,
        }
        setContextFloor(chatId, stored)
        setFloor(stored)
      }
      return found
    },
    [chatId]
  )

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
    (
      content: string,
      modelId: string,
      budget: ThinkBudget,
      attachments: Attachment[] = []
    ) => {
      setError(null)
      setIsSending(true)
      // User message + an empty assistant message we fill as tokens arrive.
      // The question carries its attachments optimistically for the same reason
      // it carries its text: the refetch on `done` replaces both with the rows
      // the server actually wrote.
      setMessages((prev) => [
        ...prev,
        {
          role: "user",
          content,
          attachments: attachments.length ? attachments : undefined,
        },
        { role: "assistant", content: "" },
      ])

      // Counts every token frame the socket delivers for this turn. A token is
      // counted as "thinking" when it falls inside a <|thinking|>…<|/thinking|>
      // block (the opening and closing markers included). We track the block
      // state from the raw token values rather than inferring it from "no
      // visible answer yet" — the model can think after it starts answering,
      // and can open several thinking blocks per reply, so the position of the
      // first visible word tells us nothing reliable about thinking length.
      //
      // The tally is advanced in `onToken` below and the running totals are
      // passed *into* these updaters. It used to be kept inside them, which
      // made the count wrong by a whole factor: a `setState` updater is not
      // allowed to have effects, because React runs it as many times as it
      // likes — twice under StrictMode, again on any render it replays — and
      // each extra run charged the same token to the counter again. Appending
      // the text is safe there (it derives from `prev`, so re-running is
      // idempotent) which is why the reply read correctly while its byline
      // claimed exactly twice the tokens the server had generated.
      let totalTokens = 0
      let thinkingTokens = 0
      let insideThinking = false

      const appendToAssistant = (
        value: string,
        tokenCount: number,
        thinkingTokenCount: number
      ) =>
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
              tokenCount,
              thinkingTokenCount,
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

      /**
       * A later round called again, so what streamed as the answer was working
       * after all. Fold it back and let the trace keep growing.
       *
       * Without this the turn reads as two chains: the trace, then a loose
       * reasoning block and a bare call underneath it — which is what the reader
       * sees the moment a model looks at a tool result and decides to try
       * something else, i.e. whenever it is actually working.
       */
      const resumeWorking = () =>
        setMessages((prev) => {
          const next = [...prev]
          const last = next[next.length - 1]
          if (last?.role === "assistant" && last.working !== undefined) {
            next[next.length - 1] = {
              ...last,
              working: joinWorking(last.working, last.content),
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
      const appendToWorking = (value: string, tokenCount: number) =>
        setMessages((prev) => {
          const next = [...prev]
          const last = next[next.length - 1]
          if (last?.role === "assistant" && last.working !== undefined) {
            const sep = last.working ? " " : ""
            next[next.length - 1] = {
              ...last,
              working: last.working + sep + value,
              tokenCount,
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

      // A stored floor was cut to fit *some* model's `ctx_window` — resending
      // it to a different one would prune this turn's history at a point
      // sized for a model that isn't the one answering it. Only trust it while
      // it's still for the model this turn is actually going to.
      const activeFloor = floorRef.current
      const historyFloor =
        activeFloor?.modelId === modelId ? activeFloor.messageId : null

      return new Promise<void>((resolve) => {
        cancelRef.current = streamChat(
          chatId,
          { text: content, attachmentIds: attachments.map((a) => a.id) },
          modelId,
          systemRef.current,
          budget,
          samplingRef.current,
          historyFloor,
          {
            onToken: (value: string) => {
              setQueuedAhead(0)
              // One frame, one token — counted once, here, where the arrival of
              // the frame is the event. See the note on the appenders above.
              totalTokens += 1
              if (working && !answering) {
                appendToWorking(value, totalTokens)
                return
              }
              if (value.includes("<|thinking|>")) insideThinking = true
              if (insideThinking) thinkingTokens += 1
              if (value.includes("<|/thinking|>")) insideThinking = false
              appendToAssistant(value, totalTokens, thinkingTokens)
            },
            onToolCall: () => {
              // The markup spelling the call has already streamed into the
              // reply, so there is nothing to add — this only marks where the
              // working phase begins.
              //
              // A call arriving while `answering` means a later round decided to
              // call rather than answer: what has streamed since the last result
              // belongs to the working, not to the answer. Only the round that
              // stops calling ends the turn.
              if (answering) {
                resumeWorking()
                answering = false
              }
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
    if (pending) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      sendMessage(
        pending.content,
        pending.modelId,
        pending.budget,
        pending.attachments
      )
    }
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
    // Tokens `/prune` has excluded from this chat's own future turns — the
    // footer subtracts this from the lifetime total it would otherwise show.
    // See `lib/context-floor.ts` for why this is a subtraction and not a
    // second server-reported total. 0 (not just absent, but for the wrong
    // model) whenever the stored floor isn't the current model's own — a
    // badge claiming tokens are excluded when nothing is actually being cut
    // from *this* model's prompt would be a worse lie than showing none.
    prunedTokens: floor?.modelId === currentModel ? floor.prunedTokens : 0,
    pruneContext,
  }
}
