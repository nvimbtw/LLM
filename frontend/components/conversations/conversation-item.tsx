"use client"

import * as React from "react"
import Link from "next/link"
import { MoreHorizontal, Pencil, Trash2 } from "lucide-react"

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
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { useChats, type ChatSummary } from "@/components/chat/chats-provider"

function formatDate(iso: string) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ""
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  })
}

export function ConversationItem({ chat }: { chat: ChatSummary }) {
  const { renameChat, deleteChat } = useChats()

  const [isRenaming, setRenaming] = React.useState(false)
  const [title, setTitle] = React.useState(chat.title)
  const [confirmOpen, setConfirmOpen] = React.useState(false)

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
  }

  return (
    <li className="flex items-center gap-2 px-4 py-3">
      {isRenaming ? (
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
          className="h-8"
        />
      ) : (
        <Link
          href={`/chat/${chat.id}`}
          className="min-w-0 flex-1 rounded-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <span className="block truncate text-sm font-medium">
            {chat.title}
          </span>
          <span className="text-xs text-muted-foreground">
            {formatDate(chat.createdAt)}
          </span>
        </Link>
      )}

      <DropdownMenu>
        <DropdownMenuTrigger
          render={
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label="Chat options"
              className="shrink-0 text-muted-foreground"
            >
              <MoreHorizontal />
            </Button>
          }
        />
        <DropdownMenuContent align="end">
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
    </li>
  )
}
