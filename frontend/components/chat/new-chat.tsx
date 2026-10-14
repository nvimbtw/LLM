"use client"

import { NewChatScene } from "./new-chat-scene"
import { ChatComposer } from "./chat-composer"
import { useStartChat } from "@/hooks/use-start-chat"
import { FormError } from "@/components/ui/error-state"

export interface NewChatProps {
  projectId?: string
  embedded?: boolean
}

/** Renders the real new-chat composer either on the home scene or inside its owning project page. */
export function NewChat({ projectId, embedded = false }: NewChatProps) {
  const start = useStartChat(projectId)
  const composer = (
    <>
      <ChatComposer
        plan={start.localPlan.plan}
        onPlanChange={start.localPlan.save}
        onSend={start.send}
        disabled={start.isSending}
        title="New chat"
        contextTokensUsed={0}
      />
      {start.error && <FormError className="mt-2">{start.error}</FormError>}
    </>
  )
  return embedded ? composer : <NewChatScene>{composer}</NewChatScene>
}
