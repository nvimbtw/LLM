"use client"

import * as React from "react"

import { hasVisibleContent } from "@/lib/message-parser"
import { useChat } from "@/hooks/use-chat"
import { ChatComposer } from "./chat-composer"

/** How close (px) to the bottom the user must be for auto-scroll to engage. */
const STICK_TO_BOTTOM_THRESHOLD_PX = 120
import { ChatMessageBubble } from "./chat-message"
import { FluxThrobber } from "./flux-throbber"
import { ThinkingLabel } from "./thinking-label"

export function ChatThread({ chatId }: { chatId: string }) {
  const { messages, isLoading, isSending, error, sendMessage } = useChat(chatId)
  const scrollRef = React.useRef<HTMLDivElement>(null)

  const lastMessage = messages[messages.length - 1]
  const showIdle =
    !isLoading && !isSending && lastMessage?.role === "assistant"
  // While streaming, show "Thinking…" until real answer text appears (the
  // <|thinking|> block is hidden, so it counts as no visible content yet).
  const streaming = isSending && lastMessage?.role === "assistant"
  // Thinking phase: no visible answer yet. Generating phase: answer streaming in.
  const showThinking = streaming && !hasVisibleContent(lastMessage.content)
  const showGenerating = streaming && hasVisibleContent(lastMessage.content)

  // Streaming mutates the last message on every token, so this effect fires
  // many times per second — scroll instantly (smooth animations overlap and
  // judder) and only when the user is already near the bottom, so scrolling
  // up to read isn't hijacked.
  React.useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    if (distanceFromBottom < STICK_TO_BOTTOM_THRESHOLD_PX) {
      el.scrollTop = el.scrollHeight
    }
  }, [messages, isSending])

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="mx-auto flex w-full max-w-2xl flex-col gap-4 px-4 py-6">
          {isLoading ? (
            <div className="flex justify-center py-8">
              <FluxThrobber mode="loading" className="h-16 w-40" />
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
            <div className="flex h-11 items-center gap-2.5 px-1 text-sm">
              {showThinking && (
                <>
                  <FluxThrobber mode="thinking" className="h-8 w-14 shrink-0" />
                  <ThinkingLabel />
                  {!!lastMessage.thinkingTokenCount && (
                    <span className="text-xs text-muted-foreground tabular-nums">
                      {lastMessage.thinkingTokenCount.toLocaleString("en-US")} tokens
                    </span>
                  )}
                </>
              )}
              {showGenerating && (
                <>
                  <FluxThrobber mode="generating" className="h-8 w-14" />
                  {!!lastMessage.tokenCount && (
                    <span className="text-xs text-muted-foreground tabular-nums">
                      {lastMessage.tokenCount.toLocaleString("en-US")} tokens
                    </span>
                  )}
                </>
              )}
              {showIdle && <FluxThrobber mode="idle" className="h-11 w-28" />}
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
