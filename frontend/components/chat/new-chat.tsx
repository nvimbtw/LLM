"use client"

import * as React from "react"
import { useRouter } from "next/navigation"

import { FigureTrellis } from "@/components/ui/figure-trellis"
import { useChats } from "./chats-provider"
import { ChatComposer } from "./chat-composer"
import type { ThinkBudget } from "@/lib/think-budget"
import type { Attachment } from "@/lib/attachments"
import { useLocalPlan } from "@/hooks/use-local-plan"
import { writePlan } from "@/lib/plan-storage"
import { toast } from "@/components/ui/toast"

export function NewChat() {
  const localPlan = useLocalPlan()
  const router = useRouter()
  const { createChat } = useChats()
  const [isSending, setIsSending] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  async function handleSend(
    content: string,
    modelId: string,
    budget: ThinkBudget,
    attachments: Attachment[]
  ) {
    setIsSending(true)
    setError(null)
    try {
      const chatId = await createChat(content, modelId, budget, attachments)
      if (localPlan.plan) {
        try {
          writePlan(chatId, localPlan.plan)
        } catch {
          toast.error(
            "Plan not saved",
            "Your chat was created, but this browser could not store its plan."
          )
        }
      }
      router.push(`/chat/${chatId}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to start a chat.")
      setIsSending(false)
    }
  }

  return (
    // No optical correction any more: <main> now starts at the true page top
    // (the 48px header this used to cancel is gone — see (app)/layout.tsx), so
    // centring on its own box and centring on the page are the same thing. The
    // old `pb-12` would now push the group 24px *high*.
    <div className="mx-auto flex w-full max-w-2xl flex-1 flex-col items-center justify-center px-4">
      {/* Only the empty state gets a figure. A conversation with messages in it
          has a subject already, and `chat-thread.tsx` stays as it was. */}
      <FigureTrellis className="max-w-md" />
      <h1 className="mt-1 font-display text-3xl font-semibold tracking-tight">
        What can I help with?
      </h1>
      <div className="mt-8 w-full">
        <ChatComposer
          plan={localPlan.plan}
          onPlanChange={localPlan.save}
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
