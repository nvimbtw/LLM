"use client"

import * as React from "react"

import { hasVisibleContent } from "@/lib/message-parser"
import { useChat } from "@/hooks/use-chat"
import { ChatComposer } from "./chat-composer"
import { ChatMessageBubble } from "./chat-message"
import { FluxThrobber } from "./flux-throbber"
import { ThinkingLabel } from "./thinking-label"

export function ChatThread({ chatId }: { chatId: string }) {
  const { messages, isLoading, isSending, error, sendMessage } = useChat(chatId)
  const bottomRef = React.useRef<HTMLDivElement>(null)

  const lastMessage = messages[messages.length - 1]
  const showIdle =
    !isLoading && !isSending && lastMessage?.role === "assistant"
  // While streaming, show "Thinking…" until real answer text appears (the
  // <|thinking|> block is hidden, so it counts as no visible content yet).
  const streaming = isSending && lastMessage?.role === "assistant"
  // Thinking phase: no visible answer yet. Generating phase: answer streaming in.
  const showThinking = streaming && !hasVisibleContent(lastMessage.content)
  const showGenerating = streaming && hasVisibleContent(lastMessage.content)

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isSending])

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto">
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

          {showThinking && (
            <div className="flex items-center gap-2.5 px-1 py-2 text-sm">
              <FluxThrobber mode="thinking" className="h-8 w-14 shrink-0" />
              <ThinkingLabel />
            </div>
          )}

          {showGenerating && (
            <div className="flex justify-start px-1 py-2">
              <FluxThrobber mode="thinking" className="h-8 w-14" />
            </div>
          )}

          {showIdle && (
            <div className="flex justify-start px-1 py-2">
              <FluxThrobber mode="idle" className="h-11 w-28" />
            </div>
          )}

          {error && <p className="text-sm text-destructive">{error}</p>}

          <div ref={bottomRef} />
        </div>
      </div>

      <div className="mx-auto w-full max-w-2xl px-4 pb-4">
        <ChatComposer onSend={sendMessage} disabled={isSending} />
      </div>
    </div>
  )
}
