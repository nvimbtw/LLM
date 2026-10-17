"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"

import {
  cachedChatHistory,
  invalidateChatHistory,
  loadChatHistory,
} from "@/lib/chat-history"
import { groupTurns, type ChatMessage } from "@/lib/chat-messages"
import {
  applyChatMessagesChange,
  ChatSession,
  type ChatMessagesChange,
  type LoadedChatHistory,
  type ChatSendOptions,
} from "@/lib/chat-session"
import {
  fetchContextFloor,
  getContextFloor,
  setContextFloor,
  type StoredFloor,
} from "@/lib/context-floor"
import type { Attachment } from "@/lib/attachments"
import { setMessageFeedback, type Feedback } from "@/lib/feedback"
import { useChats } from "@/components/chat/chats-provider"
import { useComposerChoice } from "@/components/chat/composer-choice-provider"
import { useSettings } from "@/components/settings/settings-provider"
import type { ThinkBudget } from "@/lib/think-budget"

export type { ChatMessage } from "@/lib/chat-messages"

/**
 * Loads and prunes the browser-local floor for a chat; returns the stored cut and prune operation.
 * A floor is applied only to the model whose context window sized it.
 */
function useChatContextFloor(chatId: string) {
  const [floor, setFloor] = React.useState<StoredFloor | null>(() =>
    getContextFloor(chatId)
  )
  const [loadedFor, setLoadedFor] = React.useState(chatId)
  if (loadedFor !== chatId) {
    setLoadedFor(chatId)
    setFloor(getContextFloor(chatId))
  }
  /** Computes a model-specific cut and stores it when at least one turn was excluded. */
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
  return { floor, pruneContext }
}

/**
 * Coordinates cached history, streaming, feedback and context pruning for one chat id.
 * Pure turn transformations live in lib/chat-turn; ChatSession owns socket/queue lifetimes.
 */
export function useChat(chatId: string) {
  const { takePendingMessage, refresh: refreshChats, catalog } = useChats()
  const token = useSessionToken()
  const ready = Boolean(token) && catalog.getSnapshot().token === token
  const { settings } = useSettings()
  const { model: currentModel } = useComposerChoice()
  const { floor, pruneContext } = useChatContextFloor(chatId)
  const [messages, setMessages] = React.useState<ChatMessage[]>(() =>
    groupTurns(cachedChatHistory(chatId)?.messages ?? [])
  )
  const [isLoading, setIsLoading] = React.useState(
    () => cachedChatHistory(chatId) === undefined
  )
  const [error, setError] = React.useState<string | null>(null)
  const [loadedOnce, setLoadedOnce] = React.useState(false)
  const [loadedFor, setLoadedFor] = React.useState(chatId)
  if (loadedFor !== chatId) {
    setLoadedFor(chatId)
    const cached = cachedChatHistory(chatId)
    setMessages(groupTurns(cached?.messages ?? []))
    setIsLoading(cached === undefined)
    setError(null)
    setLoadedOnce(false)
  }

  const settingsRef = React.useRef({ settings, floor })
  const messagesRef = React.useRef(messages)
  React.useEffect(() => {
    settingsRef.current = { settings, floor }
  }, [settings, floor])
  React.useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  /** Loads folded history for this chat, bypassing a prefetched entry when fresh is requested. */
  const fetchMessages = React.useCallback(
    async (fresh = false): Promise<LoadedChatHistory> => {
      const version = catalog.getVersion()
      if (fresh) invalidateChatHistory(chatId)
      const history = await loadChatHistory(chatId)
      return {
        messages: groupTurns(history.messages),
        summary: history.summary,
        version,
      }
    },
    [chatId, catalog]
  )
  /** Applies session notifications without performing effects inside React's replayable updater. */
  const changeMessages = React.useCallback(
    (change: ChatMessagesChange) => {
      if (change.type === "start") {
        invalidateChatHistory(chatId)
        setError(null)
      }
      if (change.type === "history")
        catalog.rememberSummary(change.summary, change.version)
      setMessages((previous) => applyChatMessagesChange(previous, change))
    },
    [chatId, catalog]
  )
  /** Reads current preferences at send time and applies a floor only to its matching model. */
  const sendOptions = React.useCallback((modelId: string): ChatSendOptions => {
    const { settings: current, floor: activeFloor } = settingsRef.current
    return {
      system: current.systemPrompt,
      sampling: {
        temperature: current.temperature,
        topK: current.topK,
        topP: current.topP,
        contextMultiplier: current.contextMultiplier,
      },
      historyFloor:
        activeFloor?.modelId === modelId ? activeFloor.messageId : null,
    }
  }, [])
  const session = React.useMemo(
    () =>
      new ChatSession(chatId, {
        changeMessages,
        loadMessages: () => fetchMessages(true),
        refreshChats,
      }),
    [chatId, changeMessages, fetchMessages, refreshChats]
  )
  const stream = React.useSyncExternalStore(
    session.subscribe,
    session.getSnapshot,
    session.getSnapshot
  )

  React.useEffect(() => {
    session.configure(sendOptions)
  }, [session, sendOptions])
  React.useEffect(() => {
    session.activate()
    return () => session.dispose()
  }, [session])
  React.useEffect(() => {
    if (!ready) return
    let active = true
    fetchMessages()
      .then((data) => {
        if (active) {
          changeMessages({ type: "history", ...data })
        }
      })
      .catch((failure: unknown) => {
        if (active)
          setError(
            failure instanceof Error
              ? failure.message
              : "Unable to load this chat."
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
  }, [fetchMessages, changeMessages, ready])

  /** Sends an ordinary turn or text steering; attachments are claimed only by ordinary turns. */
  const sendMessage = React.useCallback(
    (
      content: string,
      modelId: string,
      budget: ThinkBudget,
      attachments: Attachment[] = []
    ) => session.send(content, modelId, budget, attachments),
    [session]
  )
  React.useEffect(() => {
    if (!loadedOnce) return
    let active = true
    queueMicrotask(() => {
      if (!active) return
      const pending = takePendingMessage(chatId)
      if (pending)
        void sendMessage(
          pending.content,
          pending.modelId,
          pending.budget,
          pending.attachments
        )
    })
    return () => {
      active = false
    }
  }, [loadedOnce, chatId, takePendingMessage, sendMessage])

  /** Optimistically toggles a persisted reply's feedback and restores it if the write fails. */
  const rateMessage = React.useCallback(
    async (messageId: string, feedback: Feedback) => {
      const previous = messagesRef.current.find(
        (message) => message.id === messageId
      )?.feedback
      const next = previous === feedback ? undefined : feedback
      setMessages((current) =>
        current.map((message) =>
          message.id === messageId ? { ...message, feedback: next } : message
        )
      )
      try {
        invalidateChatHistory(chatId)
        await setMessageFeedback(messageId, next ?? null)
      } catch {
        setMessages((current) =>
          current.map((message) =>
            message.id === messageId
              ? { ...message, feedback: previous }
              : message
          )
        )
        setError("Your feedback could not be saved.")
      } finally {
        invalidateChatHistory(chatId)
      }
    },
    [chatId]
  )

  return {
    ...stream,
    tokenArt: session.tokenArt,
    messages,
    summary: catalog.getSummary(chatId) ?? cachedChatHistory(chatId)?.summary,
    isLoading,
    error: error ?? stream.error,
    sendMessage,
    cancel: session.cancel,
    rateMessage,
    prunedTokens: floor?.modelId === currentModel ? floor.prunedTokens : 0,
    pruneContext,
  }
}
