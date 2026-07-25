"use client"

import * as React from "react"

import { hasVisibleContent, isInThinkingBlock } from "@/lib/message-parser"
import { useChat } from "@/hooks/use-chat"
import { ChatComposer } from "./chat-composer"

/** How close (px) to the bottom the user must be for auto-scroll to engage. */
const STICK_TO_BOTTOM_THRESHOLD_PX = 120
import { ChatMessageBubble } from "./chat-message"
import { Throbber } from "@/components/ui/throbber"
import { ThinkingLabel } from "./thinking-label"

export function ChatThread({ chatId }: { chatId: string }) {
  const { messages, isLoading, isSending, error, sendMessage } = useChat(chatId)
  const scrollRef = React.useRef<HTMLDivElement>(null)
  // Whether to keep the thread pinned to the bottom. Driven by the user's own
  // wheel/touch gestures (see trackStickIntent) so our programmatic scrolls
  // never fight them, and the last measured height so we only follow real
  // growth rather than every token mutation.
  const stickRef = React.useRef(true)
  const lastHeightRef = React.useRef(0)

  const lastMessage = messages[messages.length - 1]
  const showIdle =
    !isLoading && !isSending && lastMessage?.role === "assistant"
  const streaming = isSending && lastMessage?.role === "assistant"
  // Show "Thinking…" whenever the model is *currently* inside a <|thinking|>
  // block (its content is hidden), or before any answer text has appeared at
  // all (the initial wait). Otherwise it's actively producing the answer, so
  // show "Generating…". Tracking the live block state — rather than latching on
  // "the answer has started" — means mid-reply reasoning is reflected too.
  const inThinking = streaming && isInThinkingBlock(lastMessage.content)
  const showThinking =
    streaming && (inThinking || !hasVisibleContent(lastMessage.content))
  const showGenerating = streaming && !showThinking

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
    el.scrollTo({ top: el.scrollHeight, behavior: isSending ? "smooth" : "auto" })
  }, [messages, isSending])

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div
        ref={scrollRef}
        onWheel={trackStickIntent}
        onTouchMove={trackStickIntent}
        className="flex-1 overflow-y-auto"
      >
        <div className="mx-auto flex w-full max-w-2xl flex-col gap-4 px-4 py-6">
          {isLoading ? (
            <div className="flex justify-center px-1 py-8">
              <Throbber mode="loading" className="h-5" />
            </div>
          ) : (
            messages.map((message, i) => (
              <ChatMessageBubble
                key={i}
                message={message}
                streaming={
                  isSending &&
                  i === messages.length - 1 &&
                  message.role === "assistant"
                }
              />
            ))
          )}

          {/* One fixed-height slot for all throbber states, so the
              thinking → generating → idle swaps don't reflow the thread. */}
          {(showThinking || showGenerating || showIdle) && (
            // Mark, label, then the token count pushed to the right edge. One
            // Throbber instance rather than one per state — that's what lets
            // thinking → generating → idle morph instead of swapping, since the
            // element is never unmounted.
            //
            // items-baseline, not items-center: the mark's bottom edge is its own
            // baseline (see throbber.tsx), so this sits the ticks on the same line as
            // the label instead of centring two boxes of different heights.
            <div className="flex h-11 items-baseline gap-3 px-1 text-sm">
              <Throbber
                mode={showThinking ? "thinking" : showGenerating ? "generating" : "idle"}
                className="h-5 shrink-0"
              />
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

          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>
      </div>

      <div className="mx-auto w-full max-w-2xl px-4 pb-4">
        <ChatComposer onSend={sendMessage} disabled={isSending} />
      </div>
    </div>
  )
}
