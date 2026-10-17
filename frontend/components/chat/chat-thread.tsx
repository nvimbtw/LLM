"use client"

import * as React from "react"

import { cn } from "@/lib/utils"
import { hasVisibleContent, isInThinkingBlock } from "@/lib/message-parser"
import type { Attachment } from "@/lib/attachments"
import type { Feedback } from "@/lib/feedback"
import type { ChatMessage } from "@/lib/chat-messages"
import { useSessionToken } from "@/hooks/use-auth"
import { useChat } from "@/hooks/use-chat"
import { useChatFollowing } from "@/hooks/use-chat-following"
import { useLocalPlan } from "@/hooks/use-local-plan"
import { ChatComposer } from "./chat-composer"
import { useChats } from "./chats-provider"
import { ChatPositionRail } from "./chat-position-rail"
import { useComposerChoice } from "./composer-choice-provider"

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

/** Renders one visible turn and keeps unchanged history out of token-driven renders. */
const ChatMessageRow = React.memo(function ChatMessageRow({
  message,
  streaming,
  busy,
  onRate,
  onEdit,
  onResend,
  onOpenAttachment,
}: {
  message: ChatMessage
  streaming: boolean
  busy: boolean
  onRate: (id: string, feedback: Feedback) => void
  onEdit: (content: string) => void
  onResend: (content: string) => void
  onOpenAttachment: (attachment: Attachment) => void
}) {
  const id = message.id
  return (
    <ChatMessageBubble
      message={message}
      streaming={streaming}
      onRate={
        id && message.role === "assistant"
          ? (feedback) => onRate(id, feedback)
          : undefined
      }
      onEdit={onEdit}
      onResend={onResend}
      busy={busy}
      onOpenAttachment={onOpenAttachment}
    />
  )
})

/** Keys the active conversation to its actual session, disposing sockets, drafts and private views on account replacement. */
export function ChatThread({ chatId }: ChatThreadProps) {
  const token = useSessionToken()
  return token ? (
    <SessionChatThread key={`${token}:${chatId}`} chatId={chatId} />
  ) : null
}

/** Renders a conversation with message navigation and a compact resting composer. Streaming follows growth while the reader stays near the bottom; context totals exclude a model-matching prune floor. */
function SessionChatThread({ chatId }: ChatThreadProps) {
  const localPlan = useLocalPlan(chatId)
  const {
    messages,
    summary,
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
  const recentSummary = conversations.find((c) => c.id === chatId)
  const chatSummary = summary ?? recentSummary
  const [renamedTitle, setRenamedTitle] = React.useState<string | null>(null)

  /** Persists a changed title and keeps this open thread current after its history was cached. */
  const renameCurrentChat = React.useCallback(
    async (next: string) => {
      await renameChat(chatId, next)
      setRenamedTitle(next)
    },
    [chatId, renameChat]
  )
  const [draft, setDraft] = React.useState<{ text: string } | undefined>()
  const [viewing, setViewing] = React.useState<Attachment | null>(null)
  const onRate = React.useCallback(
    (id: string, feedback: Feedback) => void rateMessage(id, feedback),
    [rateMessage]
  )
  const onEdit = React.useCallback((content: string) => {
    setDraft({ text: content })
  }, [])
  const onResend = React.useCallback(
    (content: string) => void sendMessage(content, model, budget),
    [sendMessage, model, budget]
  )
  const scrollRef = React.useRef<HTMLDivElement>(null)
  const contentRef = React.useRef<HTMLDivElement>(null)
  const composerRef = React.useRef<HTMLDivElement>(null)
  const { pauseFollowing, trackStickIntent } = useChatFollowing({
    scrollRef,
    contentRef,
    composerRef,
  })

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

  return (
    <div className="flex flex-1 overflow-hidden">
      <div
        className={cn(
          "@container/chat relative flex min-w-0 flex-1 flex-col overflow-hidden",
          viewing && "hidden lg:flex"
        )}
      >
        <ChatPositionRail
          messages={isLoading ? [] : messages}
          scrollRef={scrollRef}
          contentRef={contentRef}
          onNavigate={pauseFollowing}
        />
        <div
          ref={scrollRef}
          onWheel={trackStickIntent}
          onTouchMove={trackStickIntent}
          className="flex min-h-0 flex-1 flex-col overflow-y-auto pb-(--chat-composer-height)"
        >
          <div
            ref={contentRef}
            className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-7 px-4 py-6 sm:px-8"
          >
            {isLoading ? (
              <div className="flex justify-center px-1 py-8">
                <Throbber mode="loading" />
              </div>
            ) : (
              messages.map((message, i) => {
                return (
                  <div
                    key={i}
                    data-message-index={i}
                    data-chat-question={
                      message.role === "user" ? "true" : undefined
                    }
                  >
                    <ChatMessageRow
                      message={message}
                      streaming={
                        isSending &&
                        i === messages.length - 1 &&
                        message.role === "assistant"
                      }
                      onRate={onRate}
                      onEdit={onEdit}
                      onResend={onResend}
                      busy={isSending || !model}
                      onOpenAttachment={setViewing}
                    />
                  </div>
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
        </div>
        <div
          ref={composerRef}
          className="absolute inset-x-0 bottom-0 z-10 mx-auto w-full max-w-2xl px-4 pt-2 pb-4 sm:px-8"
        >
          <ChatComposer
            key={chatId}
            plan={localPlan.plan}
            onPlanChange={localPlan.save}
            onSend={sendMessage}
            sending={isSending}
            compact={messages.length > 0}
            onCancel={cancel}
            title={
              renamedTitle ??
              recentSummary?.title ??
              summary?.title ??
              "New chat"
            }
            onRenameTitle={(next) => void renameCurrentChat(next)}
            contextTokensUsed={contextTokensUsed}
            unseenTokens={prunedTokens}
            draft={draft}
            onPruneContext={() => pruneContext(model)}
          />
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
