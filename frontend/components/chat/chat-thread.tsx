"use client"

import * as React from "react"

import { cn } from "@/lib/utils"
import { hasVisibleContent, isInThinkingBlock } from "@/lib/message-parser"
import type { Attachment } from "@/lib/attachments"
import { useSessionToken } from "@/hooks/use-auth"
import { useChat } from "@/hooks/use-chat"
import { useLocalPlan } from "@/hooks/use-local-plan"
import { ChatComposer } from "./chat-composer"
import { useChats } from "./chats-provider"
import { ConversationProjectControl } from "@/components/projects/conversation-project"
import { useComposerChoice } from "./composer-choice-provider"

/** How close (px) to the bottom the user must be for auto-scroll to engage. */
const STICK_TO_BOTTOM_THRESHOLD_PX = 120
import { AttachmentViewer } from "./attachment-viewer"
import { ChatMessageBubble } from "./chat-message"
import { Throbber } from "@/components/ui/throbber"
import { LoadError } from "@/components/ui/error-state"
import { ThinkingLabel } from "./thinking-label"
import { callInFlight } from "./trace"
import { ToolLabel } from "./tool-label"
import { QueuedMessage } from "./queued-message"
import { TokenFlow, type TokenFlowProps } from "./token-flow"
import { TurnClock, workedFor } from "./turn-clock"

/** The join between two things in the status row. One element, reused: it is
 * punctuation, and punctuation that differs between two commas in a sentence is
 * a mistake somewhere. */
const RULE = (
  <span aria-hidden className="-mx-1 shrink-0 text-border">
    |
  </span>
)

export interface ChatThreadProps {
  chatId: string
}

/** Keys the active conversation to its actual session, disposing sockets, drafts and private views on account replacement. */
export function ChatThread({ chatId }: ChatThreadProps) {
  const token = useSessionToken()
  return token ? (
    <SessionChatThread key={`${token}:${chatId}`} chatId={chatId} />
  ) : null
}

/** Renders a conversation and its project association. Streaming follows growth only while the reader stays near the bottom; context totals include persisted and live tokens, excluding a model-matching prune floor. */
function SessionChatThread({ chatId }: ChatThreadProps) {
  const localPlan = useLocalPlan(chatId)
  const {
    messages,
    isLoading,
    isSending,
    queuedAhead,
    queued,
    error,
    sendMessage,
    cancel,
    rateMessage,
    prunedTokens,
    pruneContext,
    tokenArt,
  } = useChat(chatId)
  const { conversations, renameChat } = useChats()
  const { model, budget } = useComposerChoice()
  const chatSummary = conversations.find((c) => c.id === chatId)
  const [draft, setDraft] = React.useState<{ text: string } | undefined>()
  const [viewing, setViewing] = React.useState<Attachment | null>(null)
  const scrollRef = React.useRef<HTMLDivElement>(null)
  const stickRef = React.useRef(true)
  const lastHeightRef = React.useRef(0)

  const lastMessage = messages[messages.length - 1]
  const contextTokensUsed =
    Math.max(0, (chatSummary?.tokens ?? 0) - prunedTokens) +
    (isSending ? (lastMessage?.tokenCount ?? 0) : 0)

  const showIdle = !isLoading && !isSending && lastMessage?.role === "assistant"
  const waiting = isSending && queuedAhead !== null && queuedAhead > 0
  const streaming = isSending && !waiting && lastMessage?.role === "assistant"
  const call = streaming
    ? callInFlight(lastMessage.working, lastMessage.toolAnswers ?? [])
    : null
  const searching = call !== null
  const inThinking = streaming && isInThinkingBlock(lastMessage.content)
  const showThinking =
    streaming &&
    !searching &&
    (inThinking || !hasVisibleContent(lastMessage.content))
  const showGenerating = streaming && !searching && !showThinking
  const label = searching || waiting || showThinking
  const written = isSending ? (lastMessage?.tokenCount ?? 0) : 0
  const statusFlow: TokenFlowProps | null =
    contextTokensUsed > 0 && !showIdle
      ? {
          direction: waiting || searching || written === 0 ? "in" : "out",
          count: contextTokensUsed,
        }
      : null
  /** Updates bottom-following intent from a real wheel or touch gesture. */
  const trackStickIntent = React.useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    stickRef.current = distanceFromBottom < STICK_TO_BOTTOM_THRESHOLD_PX
  }, [])
  React.useEffect(() => {
    const el = scrollRef.current
    if (!el || !stickRef.current) return
    const grew = el.scrollHeight > lastHeightRef.current
    lastHeightRef.current = el.scrollHeight
    if (!grew) return
    el.scrollTo({ top: el.scrollHeight, behavior: "auto" })
  }, [messages, isSending])

  return (
    <div className="flex flex-1 overflow-hidden">
      <div
        className={cn(
          "flex min-w-0 flex-1 flex-col overflow-hidden",
          viewing && "hidden lg:flex"
        )}
      >
        {chatSummary && (
          <ConversationProjectControl key={chatId} chat={chatSummary} />
        )}
        <div
          ref={scrollRef}
          onWheel={trackStickIntent}
          onTouchMove={trackStickIntent}
          className="flex flex-1 flex-col overflow-y-auto"
        >
          <div className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-7 px-4 py-8">
            {isLoading ? (
              <div className="flex justify-center px-1 py-8">
                <Throbber mode="loading" />
              </div>
            ) : (
              messages.map((message, i) => {
                const id = message.id
                return (
                  <ChatMessageBubble
                    key={i}
                    message={message}
                    streaming={
                      isSending &&
                      i === messages.length - 1 &&
                      message.role === "assistant"
                    }
                    onRate={
                      id && message.role === "assistant"
                        ? (feedback) => void rateMessage(id, feedback)
                        : undefined
                    }
                    onEdit={(content) => setDraft({ text: content })}
                    onResend={(content) =>
                      void sendMessage(content, model, budget)
                    }
                    busy={isSending || !model}
                    onOpenAttachment={setViewing}
                  />
                )
              })
            )}
            {queued.map((message) => (
              <QueuedMessage key={message.id} text={message.content} />
            ))}
            {(showThinking ||
              showGenerating ||
              searching ||
              showIdle ||
              waiting) && (
              <div className="flex items-center gap-3 px-1 text-sm">
                <Throbber
                  stream={tokenArt}
                  mode={
                    waiting
                      ? "loading"
                      : searching
                        ? "tool"
                        : showThinking
                          ? "thinking"
                          : showGenerating
                            ? "generating"
                            : "idle"
                  }
                />
                {isSending && (
                  <TurnClock key={`${chatId}:${messages.length}`} />
                )}
                {showIdle && lastMessage?.elapsedSecs != null && (
                  <span className="shrink-0 text-xs text-muted-foreground">
                    {workedFor(lastMessage.elapsedSecs)}
                  </span>
                )}
                {isSending && statusFlow && RULE}
                {statusFlow && <TokenFlow {...statusFlow} />}
                {statusFlow && label && RULE}
                {call && <ToolLabel call={call} />}
                {waiting && (
                  <span className="text-muted-foreground">
                    Queued — {queuedAhead} {queuedAhead === 1 ? "job" : "jobs"}{" "}
                    ahead on the GPU. You can close this tab; the reply will be
                    here.
                  </span>
                )}
                {showThinking && <ThinkingLabel />}
                <span className="min-w-0 flex-1" />
              </div>
            )}

            {error && <LoadError title={error} />}
          </div>
          <div className="sticky bottom-0 z-10 mx-auto w-full max-w-2xl px-4 pb-4">
            <ChatComposer
              key={chatId}
              plan={localPlan.plan}
              onPlanChange={localPlan.save}
              onSend={sendMessage}
              sending={isSending}
              onCancel={cancel}
              title={chatSummary?.title ?? "New chat"}
              onRenameTitle={(next) => void renameChat(chatId, next)}
              contextTokensUsed={contextTokensUsed}
              unseenTokens={prunedTokens}
              draft={draft}
              onPruneContext={() => pruneContext(model)}
            />
          </div>
        </div>
      </div>

      {viewing && (
        <AttachmentViewer
          key={viewing.id}
          attachment={viewing}
          onClose={() => setViewing(null)}
        />
      )}
    </div>
  )
}
