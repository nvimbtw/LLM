"use client"

import * as React from "react"
import { useRouter } from "next/navigation"

import { useChats } from "./chats-provider"
import { ChatComposer } from "./chat-composer"

export function NewChat() {
  const router = useRouter()
  const { createChat } = useChats()
  const [isSending, setIsSending] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  async function handleSend(content: string, modelId: string) {
    setIsSending(true)
    setError(null)
    try {
      const chatId = await createChat(content, modelId)
      router.push(`/chat/${chatId}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to start a chat.")
      setIsSending(false)
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-1 flex-col items-center justify-center gap-8 px-4">
      <h1 className="text-3xl font-semibold tracking-tight">
        What can I help with?
      </h1>
      <div className="w-full">
        <ChatComposer onSend={handleSend} disabled={isSending} />
        {error && <p className="mt-2 text-sm text-destructive">{error}</p>}
      </div>
    </div>
  )
}
