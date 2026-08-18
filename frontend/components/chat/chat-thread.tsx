"use client"

import * as React from "react"

import { hasVisibleContent, isInThinkingBlock } from "@/lib/message-parser"
import { useChat } from "@/hooks/use-chat"
import { ChatComposer } from "./chat-composer"
import { useChats } from "./chats-provider"

/** How close (px) to the bottom the user must be for auto-scroll to engage. */
const STICK_TO_BOTTOM_THRESHOLD_PX = 120
import { ChatMessageBubble } from "./chat-message"
import { Throbber } from "@/components/ui/throbber"
import { LoadError } from "@/components/ui/error-state"
import { ThinkingLabel } from "./thinking-label"
import { isCalling } from "./trace"

export function ChatThread({ chatId }: { chatId: string }) {
  const {
    messages,
    isLoading,
    isSending,
    queuedAhead,
    error,
    sendMessage,
    cancel,
    rateMessage,
  } = useChat(chatId)
  const { conversations, renameChat } = useChats()
  const chatSummary = conversations.find((c) => c.id === chatId)
  const scrollRef = React.useRef<HTMLDivElement>(null)
  // Whether to keep the thread pinned to the bottom. Driven by the user's own
  // wheel/touch gestures (see trackStickIntent) so our programmatic scrolls
  // never fight them, and the last measured height so we only follow real
  // growth rather than every token mutation.
  const stickRef = React.useRef(true)
  const lastHeightRef = React.useRef(0)

  const lastMessage = messages[messages.length - 1]

  // The sidebar's count is the last *persisted* total — it doesn't move until
  // this turn is saved server-side, so a turn in flight is added on top from
  // what's been tallied client-side as it streams (see use-chat.ts). Neither
  // half is server-verified against the tokenizer, so this is an estimate for
  // the composer's status row, not an accounting figure.
  const contextTokensUsed =
    (chatSummary?.tokens ?? 0) + (isSending ? (lastMessage?.tokenCount ?? 0) : 0)

  const showIdle = !isLoading && !isSending && lastMessage?.role === "assistant"
  // Waiting in line is not the same as generating: no tokens are coming yet, and
  // the reason is the queue rather than a slow model. It gets its own state so the
  // thread can say which, instead of showing a "generating" mark for a turn that
  // hasn't reached the GPU.
  const waiting = isSending && queuedAhead !== null && queuedAhead > 0
  const streaming = isSending && !waiting && lastMessage?.role === "assistant"
  // Show "Thinking…" whenever the model is *currently* inside a <|thinking|>
  // block (its content is hidden), or before any answer text has appeared at
  // all (the initial wait). Otherwise it's actively producing the answer, so
  // show "Generating…". Tracking the live block state — rather than latching on
  // "the answer has started" — means mid-reply reasoning is reflected too.
  //
  // A call in flight outranks both. It is the one phase where the wait is not
  // the model's at all — nothing is being generated while a search runs — so
  // saying "Generating…" over it would be a plain lie.
  const searching =
    streaming && isCalling(lastMessage.working, lastMessage.toolAnswers ?? [])
  const inThinking = streaming && isInThinkingBlock(lastMessage.content)
  const showThinking =
    streaming &&
    !searching &&
    (inThinking || !hasVisibleContent(lastMessage.content))
  const showGenerating = streaming && !searching && !showThinking

  // Whichever count belongs to the state on show, so the leader rule's right-hand
  // figure tracks the phase without the row having to branch.
  const statusTokens = showThinking
    ? (lastMessage.thinkingTokenCount ?? null)
    : showGenerating
      ? (lastMessage.tokenCount ?? null)
      : null

  // A user gesture near the bottom (re-)engages auto-follow; scrolling up to
  // read disengages it. Reading live scrollTop here — not during our own
  // programmatic scroll — is what keeps the two from fighting.
  const trackStickIntent = React.useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    stickRef.current = distanceFromBottom < STICK_TO_BOTTOM_THRESHOLD_PX
  }, [])

  // New chat: snap back to following, and forget the previous thread's height
  // so the first render of this one counts as growth (see below).
  React.useEffect(() => {
    stickRef.current = true
    lastHeightRef.current = 0
  }, [chatId])

  // Follow the bottom as content arrives — but only on genuine growth (a new
  // line, not an in-place token edit) and only while following. Streaming
  // glides smoothly; history loads and freshly-sent messages snap instantly.
  React.useEffect(() => {
    const el = scrollRef.current
    if (!el || !stickRef.current) return
    const grew = el.scrollHeight > lastHeightRef.current
    lastHeightRef.current = el.scrollHeight
    if (!grew) return
    el.scrollTo({
      top: el.scrollHeight,
      behavior: isSending ? "smooth" : "auto",
    })
  }, [messages, isSending])

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div
        ref={scrollRef}
        onWheel={trackStickIntent}
        onTouchMove={trackStickIntent}
        className="flex-1 overflow-y-auto"
      >
        {/* gap-7 rather than gap-4: replies are now set at 16px with a 1.65 line,
            and at that size four units between turns reads as one continuous
            column of text instead of an exchange. The air is what separates who
            is speaking, since the assistant side has no bubble to do it. */}
        <div className="mx-auto flex w-full max-w-2xl flex-col gap-7 px-4 py-8">
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
                  // Only assistant replies that exist server-side are rateable —
                  // there is nothing to attach a verdict to otherwise.
                  onRate={
                    id && message.role === "assistant"
                      ? (feedback) => void rateMessage(id, feedback)
                      : undefined
                  }
                />
              )
            })
          )}

          {/* One fixed-height slot for all throbber states, so the
              thinking → generating → idle swaps don't reflow the thread. */}
          {(showThinking || showGenerating || searching || showIdle || waiting) && (
            // Mark, label, then the token count pushed to the right edge. One
            // Throbber instance rather than one per state — that's what lets
            // thinking → generating → idle morph instead of swapping, since the
            // element is never unmounted.
            //
            // items-center, not items-baseline: the mark is a ball, so its
            // optical centre is its middle rather than its bottom edge, and a
            // baseline row would hang it below the line of the label.
            <div className="flex items-center gap-3 px-1 text-sm">
              <Throbber
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
              {/* A fixed word rather than the rotating ThinkingLabel: that one
                  guesses at what a model might be doing, and here we know. */}
              {searching && (
                <span className="text-shimmer font-medium">Searching…</span>
              )}
              {waiting && (
                <span className="text-muted-foreground">
                  Queued — {queuedAhead} {queuedAhead === 1 ? "job" : "jobs"}{" "}
                  ahead on the GPU. You can close this tab; the reply will be
                  here.
                </span>
              )}
              {showThinking && <ThinkingLabel />}
              {/* Pushes the token count to the far edge, which the throbber used to
                  do by spanning the gap itself. */}
              <span className="min-w-0 flex-1" />
              {statusTokens !== null && (
                <span className="shrink-0 text-xs text-muted-foreground tabular-nums">
                  {statusTokens.toLocaleString("en-US")} tokens
                </span>
              )}
            </div>
          )}

          {error && <LoadError title={error} />}
        </div>

        {/* Sticky, and *inside* the scroller rather than a sibling below it, so
            the thread passes behind the composer as you scroll back through it
            instead of being clipped at a hard edge. Sticky rather than absolute
            on purpose: the composer still takes real space at the end of the
            flow, so the last reply clears it at the bottom of the thread without
            anyone maintaining a padding constant against a box whose height
            changes with attachments and wrapped lines. */}
        <div className="sticky bottom-0 z-10 mx-auto w-full max-w-2xl px-4 pb-4">
          <ChatComposer
            onSend={sendMessage}
            disabled={isSending}
            onCancel={cancel}
            title={chatSummary?.title ?? "New chat"}
            onRenameTitle={(next) => void renameChat(chatId, next)}
            contextTokensUsed={contextTokensUsed}
          />
        </div>
      </div>
    </div>
  )
}
