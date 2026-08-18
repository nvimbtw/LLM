"use client"

import * as React from "react"
import { useRouter } from "next/navigation"

import { FigureHorizon } from "@/components/ui/figure-horizon"
import { useChats } from "./chats-provider"
import { ChatComposer } from "./chat-composer"
import type { ThinkBudget } from "./budget-selector"

export function NewChat() {
  const router = useRouter()
  const { createChat } = useChats()
  const [isSending, setIsSending] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  async function handleSend(
    content: string,
    modelId: string,
    budget: ThinkBudget
  ) {
    setIsSending(true)
    setError(null)
    try {
      const chatId = await createChat(content, modelId, budget)
      router.push(`/chat/${chatId}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to start a chat.")
      setIsSending(false)
    }
  }

  return (
    // pb-12 biases the centered group upward by half its value (24px) — this
    // sits inside <main>, which itself starts 48px down from the true page
    // top (the h-12 header in (app)/layout.tsx), so centering on <main>'s own
    // box alone lands 24px low. The padding cancels exactly that, so the
    // group centers on the page, not just on the box it happens to be in.
    <div className="mx-auto flex w-full max-w-2xl flex-1 flex-col items-center justify-center px-4 pb-12">
      {/* Only the empty state gets a figure. A conversation with messages in it
          has a subject already, and `chat-thread.tsx` stays as it was. */}
      <FigureHorizon className="max-w-md" />
      <h1 className="mt-1 font-display text-3xl font-semibold tracking-tight">
        What can I help with?
      </h1>
      <div className="mt-8 w-full">
        <ChatComposer
          onSend={handleSend}
          disabled={isSending}
          title="New chat"
          contextTokensUsed={0}
        />
        {error && <p className="mt-2 text-sm text-destructive">{error}</p>}
      </div>
    </div>
  )
}
