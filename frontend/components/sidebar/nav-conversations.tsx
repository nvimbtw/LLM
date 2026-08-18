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
  SidebarMenuButton,
  SidebarMenuAction,
} from "@/components/ui/sidebar"
import { Skeleton } from "@/components/ui/skeleton"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Input } from "@/components/ui/input"
import { useChats, type ChatSummary } from "@/components/chat/chats-provider"
import { cn } from "@/lib/utils"

// Keep the sidebar list short — only the most recent chats. The rest live on the
// full /conversations page (reachable from NavMain when collapsed).
const RECENT_LIMIT = 5

// Compact, editorial relative time: "just now", "12m", "3h", "2d", then a date.
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
        className="w-full justify-between text-xs font-medium tracking-[0.16em] text-sidebar-foreground/75 uppercase hover:text-sidebar-foreground"
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

function ConversationItem({ chat }: { chat: ChatSummary }) {
  const pathname = usePathname()
  const router = useRouter()
  const { renameChat, deleteChat } = useChats()

  const [isRenaming, setRenaming] = React.useState(false)
  const [title, setTitle] = React.useState(chat.title)
  const [confirmOpen, setConfirmOpen] = React.useState(false)

  const isActive = pathname === `/chat/${chat.id}`

  async function commitRename() {
    const next = title.trim()
    setRenaming(false)
    if (next && next !== chat.title) {
      await renameChat(chat.id, next)
    } else {
      setTitle(chat.title)
    }
  }

  async function handleDelete() {
    await deleteChat(chat.id)
    setConfirmOpen(false)
    if (isActive) router.push("/")
  }

  if (isRenaming) {
    return (
      <SidebarMenuItem>
        <Input
          autoFocus
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          onBlur={commitRename}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault()
              commitRename()
            } else if (e.key === "Escape") {
              setTitle(chat.title)
              setRenaming(false)
            }
          }}
          className="h-8 text-sm"
        />
      </SidebarMenuItem>
    )
  }

  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        isActive={isActive}
        render={<Link href={`/chat/${chat.id}`} />}
        className={cn(
          // One line, the same 32px as every other row in the sidebar, with a left
          // marginalia rule that grows on hover and turns to the accent when active.
          "relative pl-3.5",
          "before:absolute before:top-1/2 before:left-1 before:h-0 before:w-px before:-translate-y-1/2",
          "before:rounded-full before:bg-sidebar-foreground/30 before:transition-all before:content-['']",
          "hover:before:h-4",
          isActive &&
            "before:h-4 before:w-0.5 before:bg-primary hover:before:bg-primary"
        )}
      >
        <span className="min-w-0 flex-1 truncate">{chat.title}</span>
        {/* The age, set as a quiet trailing figure rather than a caption line of
            its own — it is what orders the list, so it earns the corner, but it is
            not a second thing to read. It fades as the row is hovered because the
            ⋯ action lands in exactly this spot: one corner, one meaning at a time,
            and no reserved gutter sitting empty the rest of the time. */}
        <span className="shrink-0 text-xs text-sidebar-foreground/45 tabular-nums transition-opacity group-focus-within/menu-item:opacity-0 group-hover/menu-item:opacity-0">
          {formatRelative(chat.createdAt)}
        </span>
      </SidebarMenuButton>

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

      <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this chat?</AlertDialogTitle>
            <AlertDialogDescription>
              &ldquo;{chat.title}&rdquo; and all its messages will be
              permanently deleted. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              className="bg-destructive text-white hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </SidebarMenuItem>
  )
}
