"use client"

import Link from "next/link"
import { useAuth } from "@/hooks/use-auth"
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
  const { isAuthenticated, isLoading } = useAuth()
  if (isLoading) return null
  if (!isAuthenticated)
    return (
      <NewChatScene>
        <div className="text-center text-sm">
          <Link href="/login" className="text-primary hover:underline">
            Log in to chat
          </Link>
          <span className="mx-3 text-muted-foreground">or</span>
          <Link
            href="/models"
            className="text-muted-foreground hover:text-foreground"
          >
            Browse published models
          </Link>
        </div>
      </NewChatScene>
    )
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
