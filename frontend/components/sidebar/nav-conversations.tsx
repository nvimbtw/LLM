"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { ChevronRight, MoreHorizontal, Pencil, Trash2 } from "lucide-react"

import {
  SidebarGroup,
  SidebarGroupLabel,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuAction,
} from "@/components/ui/sidebar"
import { Skeleton } from "@/components/ui/skeleton"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { DeleteConfirmation } from "@/components/ui/delete-confirmation"
import { ConversationRename } from "@/components/chat/conversation-rename"
import { useChats, type ChatSummary } from "@/components/chat/chats-provider"
import { prefetchChatHistory } from "@/lib/chat-history"
import { SidebarEntryButton } from "./sidebar-entry-button"

// Keep the sidebar list short — only the most recent chats. The rest live on the
// full /conversations page (reachable from NavMain when collapsed).
const RECENT_LIMIT = 5

/** Formats sidebar recency as a short age, using a date after one week and omitting invalid values. */
function formatRelative(iso: string): string {
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return ""
  const secs = Math.max(0, (Date.now() - then) / 1000)
  if (secs < 60) return "just now"
  const mins = Math.floor(secs / 60)
  if (mins < 60) return `${mins}m`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h`
  const days = Math.floor(hours / 24)
  if (days < 7) return `${days}d`
  return new Date(then).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  })
}

// Token counts are deliberately not here. They are a property worth comparing
// across a whole archive, which is what /conversations is for; in a five-row
// recency list they were a second column nobody scans, and carrying them cost
// every row a caption line and half again its height.

/** Renders recent conversations across all project memberships, with the full archive available at /conversations. */
export function NavConversations() {
  const { conversations, isLoading } = useChats()
  const [expanded, setExpanded] = React.useState(true)

  const recent = conversations.slice(0, RECENT_LIMIT)

  return (
    <SidebarGroup className="group-data-[collapsible=icon]:hidden">
      <SidebarGroupLabel
        render={
          <button
            type="button"
            aria-expanded={expanded}
            onClick={() => setExpanded((v) => !v)}
          />
        }
        className="w-full justify-between eyebrow text-sidebar-foreground/75 hover:text-sidebar-foreground"
      >
        Conversations
        <ChevronRight
          className={`size-3.5 opacity-60 transition-transform duration-200 ${expanded ? "rotate-90" : ""}`}
        />
      </SidebarGroupLabel>
      {expanded && (
        <SidebarGroupContent>
          <SidebarMenu>
            {isLoading
              ? ["72%", "58%", "84%", "65%"].map((width, i) => (
                  <SidebarMenuItem key={i}>
                    <div className="flex h-8 items-center px-2">
                      <Skeleton className="h-4" style={{ width }} />
                    </div>
                  </SidebarMenuItem>
                ))
              : recent.map((chat) => (
                  <ConversationItem key={chat.id} chat={chat} />
                ))}
          </SidebarMenu>
        </SidebarGroupContent>
      )}
    </SidebarGroup>
  )
}

/** Renders a sidebar chat with prefetch, rename and delete actions, independent of project membership. */
function ConversationItem({ chat }: { chat: ChatSummary }) {
  const pathname = usePathname()
  const router = useRouter()
  const { renameChat, deleteChat } = useChats()

  const [isRenaming, setRenaming] = React.useState(false)
  const [confirmOpen, setConfirmOpen] = React.useState(false)

  const isActive = pathname === `/chat/${chat.id}`

  /** Warms the selected chat route and account-scoped history before navigation. */
  function prefetch() {
    if (isActive) return
    router.prefetch(`/chat/${chat.id}`)
    prefetchChatHistory(chat.id)
  }

  if (isRenaming) {
    return (
      <SidebarMenuItem>
        <ConversationRename
          title={chat.title}
          onSave={(next) => renameChat(chat.id, next)}
          onDone={() => setRenaming(false)}
        />
      </SidebarMenuItem>
    )
  }

  return (
    <SidebarMenuItem>
      <SidebarEntryButton
        isActive={isActive}
        render={
          <Link
            href={`/chat/${chat.id}`}
            onMouseEnter={prefetch}
            onFocus={prefetch}
          />
        }
      >
        <span className="min-w-0 flex-1 truncate">{chat.title}</span>
        <span className="shrink-0 text-xs text-sidebar-foreground/45 tabular-nums transition-opacity group-focus-within/menu-item:opacity-0 group-hover/menu-item:opacity-0">
          {formatRelative(chat.createdAt)}
        </span>
      </SidebarEntryButton>

      <DropdownMenu>
        <DropdownMenuTrigger
          render={
            <SidebarMenuAction showOnHover aria-label="Chat options">
              <MoreHorizontal />
            </SidebarMenuAction>
          }
        />
        <DropdownMenuContent side="right" align="start">
          <DropdownMenuItem onClick={() => setRenaming(true)}>
            <Pencil />
            Rename
          </DropdownMenuItem>
          <DropdownMenuItem
            variant="destructive"
            onClick={() => setConfirmOpen(true)}
          >
            <Trash2 />
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      {confirmOpen && (
        <DeleteConfirmation
          open
          onOpenChange={setConfirmOpen}
          title="Delete this conversation?"
          description={`“${chat.title}” and all its messages will be permanently deleted. This cannot be undone.`}
          label="Delete conversation"
          onDelete={() => deleteChat(chat.id)}
          onDeleted={() => {
            if (isActive) router.push("/")
          }}
        />
      )}
    </SidebarMenuItem>
  )
}
