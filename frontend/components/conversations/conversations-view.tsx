"use client"

import { MessageSquarePlus } from "lucide-react"

import { ButtonLink } from "@/components/ui/button-link"
import { Skeleton } from "@/components/ui/skeleton"
import { useChats } from "@/components/chat/chats-provider"
import { LoadError } from "@/components/ui/error-state"
import { ConversationItem } from "./conversation-item"

// Compact token count: 812, 1.2k, 48k.
function formatTokens(n: number): string {
  if (n < 1000) return String(n)
  const k = n / 1000
  return `${k >= 10 ? Math.round(k) : k.toFixed(1)}k`
}

function Masthead({ count, tokens }: { count: number; tokens: number }) {
  return (
    <header className="mb-2 flex items-end justify-between gap-6 border-b border-border pb-6">
      <div>
        <h1 className="font-display text-3xl font-semibold tracking-tight">
          Conversations
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          The full archive of your chats.
        </p>
      </div>
      {count > 0 && (
        <dl className="flex shrink-0 gap-6 text-right tabular-nums">
          <Stat label="Threads" value={String(count)} />
          <Stat label="Tokens" value={formatTokens(tokens)} />
        </dl>
      )}
    </header>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dd className="font-display text-2xl leading-none font-semibold tracking-tight">
        {value}
      </dd>
      <dt className="mt-1.5 text-xs font-medium tracking-[0.16em] text-muted-foreground uppercase">
        {label}
      </dt>
    </div>
  )
}

export function ConversationsView() {
  const { conversations, isLoading, error } = useChats()

  if (isLoading) {
    return (
      <div>
        <Masthead count={0} tokens={0} />
        <ul>
          {Array.from({ length: 6 }).map((_, i) => (
            <li
              key={i}
              className="flex items-baseline gap-5 border-b border-border py-5 pl-4"
            >
              <Skeleton className="h-3 w-4" />
              <div className="flex-1 space-y-2">
                <Skeleton className="h-4 w-56" />
                <Skeleton className="h-2.5 w-28" />
              </div>
            </li>
          ))}
        </ul>
      </div>
    )
  }

  if (error) {
    return (
      <div>
        <Masthead count={0} tokens={0} />
        <LoadError title={error} />
      </div>
    )
  }

  if (conversations.length === 0) {
    return (
      <div>
        <Masthead count={0} tokens={0} />
        <div className="flex flex-col items-center gap-5 py-24 text-center">
          <p className="max-w-xs text-sm text-muted-foreground">
            No conversations yet. Your archive will fill in as you chat.
          </p>
          <ButtonLink variant="outline" href="/">
            <MessageSquarePlus />
            Start a chat
          </ButtonLink>
        </div>
      </div>
    )
  }

  const totalTokens = conversations.reduce((sum, c) => sum + c.tokens, 0)

  return (
    <div>
      <Masthead count={conversations.length} tokens={totalTokens} />
      <ol>
        {conversations.map((chat, i) => (
          <ConversationItem key={chat.id} chat={chat} index={i} />
        ))}
      </ol>
    </div>
  )
}
