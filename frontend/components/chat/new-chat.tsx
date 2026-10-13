"use client"

import * as React from "react"
import { useRouter } from "next/navigation"

import { NewChatScene } from "./new-chat-scene"
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
    <NewChatScene>
      <ChatComposer
        plan={localPlan.plan}
        onPlanChange={localPlan.save}
        onSend={handleSend}
        disabled={isSending}
        title="New chat"
        contextTokensUsed={0}
      />
      {error && <p className="mt-2 text-sm text-destructive">{error}</p>}
    </NewChatScene>
  )
}
