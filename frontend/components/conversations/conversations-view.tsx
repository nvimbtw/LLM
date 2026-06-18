"use client"

import Link from "next/link"
import { MessageSquarePlus } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { useChats } from "@/components/chat/chats-provider"
import { ConversationItem } from "./conversation-item"

export function ConversationsView() {
  const { conversations, isLoading, error } = useChats()

  if (isLoading) {
    return (
      <ul className="divide-y rounded-lg border">
        {Array.from({ length: 5 }).map((_, i) => (
          <li key={i} className="flex flex-col gap-2 px-4 py-3">
            <Skeleton className="h-4 w-48" />
            <Skeleton className="h-3 w-24" />
          </li>
        ))}
      </ul>
    )
  }

  if (error) {
    return <p className="text-sm text-destructive">{error}</p>
  }

  if (conversations.length === 0) {
    return (
      <div className="flex flex-col items-center gap-4 rounded-lg border border-dashed py-16 text-center">
        <p className="text-sm text-muted-foreground">No conversations yet.</p>
        <Button render={<Link href="/" />}>
          <MessageSquarePlus />
          Start a chat
        </Button>
      </div>
    )
  }

  return (
    <ul className="divide-y rounded-lg border">
      {conversations.map((chat) => (
        <ConversationItem key={chat.id} chat={chat} />
      ))}
    </ul>
  )
}
