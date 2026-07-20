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

// Keep the sidebar list short — only the most recent chats. The rest live on the
// full /conversations page (reachable from NavMain when collapsed).
const RECENT_LIMIT = 5

export function NavConversations() {
  const { conversations, isLoading } = useChats()
  const [expanded, setExpanded] = React.useState(true)

  const recent = conversations.slice(0, RECENT_LIMIT)

  return (
    <SidebarGroup className="group-data-[collapsible=icon]:hidden">
      <SidebarGroupLabel
        render={
          <button type="button" aria-expanded={expanded} onClick={() => setExpanded((v) => !v)} />
        }
        className="w-full justify-between hover:text-sidebar-foreground"
      >
        Conversations
        <ChevronRight
          className={`transition-transform duration-200 ${expanded ? "rotate-90" : ""}`}
        />
      </SidebarGroupLabel>
      {expanded && (
        <SidebarGroupContent>
          <SidebarMenu>
            {isLoading ? (
              ["72%", "58%", "84%", "65%"].map((width, i) => (
                <SidebarMenuItem key={i}>
                  <div className="flex h-8 items-center px-2">
                    <Skeleton className="h-4" style={{ width }} />
                  </div>
                </SidebarMenuItem>
              ))
            ) : (
              recent.map((chat) => <ConversationItem key={chat.id} chat={chat} />)
            )}
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
      <SidebarMenuButton isActive={isActive} render={<Link href={`/chat/${chat.id}`} />}>
        <span>{chat.title}</span>
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
          <DropdownMenuItem variant="destructive" onClick={() => setConfirmOpen(true)}>
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
              &ldquo;{chat.title}&rdquo; and all its messages will be permanently
              deleted. This cannot be undone.
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
