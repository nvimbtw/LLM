"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { useChats } from "@/components/chat/chats-provider"
import { useProjects } from "@/components/projects/projects-provider"
import { useLocalPlan } from "./use-local-plan"
import { useSessionToken } from "./use-auth"
import { getToken } from "@/lib/api"
import { writePlan } from "@/lib/plan-storage"
import type { Attachment } from "@/lib/attachments"
import type { ThinkBudget } from "@/lib/think-budget"
import { toast } from "@/components/ui/toast"

/** Owns creation from the real composer, preserving plans, uploaded attachments and send choices for ordinary or project chats. */
export function useStartChat(projectId?: string) {
  const localPlan = useLocalPlan(projectId ? `project:${projectId}` : undefined)
  const router = useRouter()
  const token = useSessionToken()
  const { createChat } = useChats()
  const { refresh: refreshProjects } = useProjects()
  const [state, setState] = React.useState<{
    token: string | null
    sending: boolean
    error: string | null
  }>({ token, sending: false, error: null })
  const active = React.useRef(true)
  const sending = React.useRef(false)
  React.useEffect(() => {
    active.current = true
    return () => {
      active.current = false
    }
  }, [])

  /** Creates one chat row and navigates only while the originating composer/account remains active. */
  const send = async (
    content: string,
    modelId: string,
    budget: ThinkBudget,
    attachments: Attachment[]
  ): Promise<void> => {
    if (!token || getToken() !== token) {
      router.push("/login")
      return
    }
    if (sending.current) return
    sending.current = true
    setState({ token, sending: true, error: null })
    try {
      const chatId = await createChat(
        content,
        modelId,
        budget,
        attachments,
        projectId
      )
      if (!active.current || getToken() !== token) return
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
      if (projectId) void refreshProjects()
      router.push(`/chat/${chatId}`)
    } catch (failure) {
      if (active.current && getToken() === token)
        setState({
          token,
          sending: false,
          error:
            failure instanceof Error
              ? failure.message
              : "Unable to start a chat.",
        })
    } finally {
      sending.current = false
    }
  }
  return {
    localPlan,
    send,
    isSending: state.token === token && state.sending,
    error: state.token === token ? state.error : null,
  }
}
