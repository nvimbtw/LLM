"use client"

import { MessageSquarePlus } from "lucide-react"

import { ButtonLink } from "@/components/ui/button-link"
import { PageHeader } from "@/components/ui/page-header"
import { Skeleton } from "@/components/ui/skeleton"
import { useChats } from "@/components/chat/chats-provider"
import { LoadError } from "@/components/ui/error-state"
import { ConversationItem } from "./conversation-item"

/** Formats archive token totals as a compact count such as 812, 1.2k or 48k. */
function formatTokens(n: number): string {
  if (n < 1000) return String(n)
  const k = n / 1000
  return `${k >= 10 ? Math.round(k) : k.toFixed(1)}k`
}

/** Composes the shared page header with archive totals and the new-conversation action. */
function ArchiveHeader({ count, tokens }: { count: number; tokens: number }) {
  return (
    <PageHeader
      title="Conversations"
      description={
        <>
          <p>The full archive of your chats.</p>
          {count > 0 && (
            <dl className="flex gap-6 text-right tabular-nums">
              <Stat label="Threads" value={String(count)} />
              <Stat label="Tokens" value={formatTokens(tokens)} />
            </dl>
          )}
        </>
      }
      action={
        <ButtonLink href="/">
          <MessageSquarePlus />
          New conversation
        </ButtonLink>
      }
    />
  )
}

/** Displays an archive total with its quiet label. */
function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dd className="font-display text-2xl leading-none font-semibold tracking-tight">
        {value}
      </dd>
      <dt className="mt-1.5 eyebrow text-muted-foreground">{label}</dt>
    </div>
  )
}

/** Renders the current account’s archive with loading, failure and empty states. */
export function ConversationsView() {
  const { conversations, isLoading, error } = useChats()

  if (isLoading) {
    return (
      <div>
        <ArchiveHeader count={0} tokens={0} />
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
        <ArchiveHeader count={0} tokens={0} />
        <LoadError title={error} />
      </div>
    )
  }

  if (conversations.length === 0) {
    return (
      <div>
        <ArchiveHeader count={0} tokens={0} />
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
      <ArchiveHeader count={conversations.length} tokens={totalTokens} />
      <ol>
        {conversations.map((chat, i) => (
          <ConversationItem key={chat.id} chat={chat} index={i} />
        ))}
      </ol>
    </div>
  )
}
