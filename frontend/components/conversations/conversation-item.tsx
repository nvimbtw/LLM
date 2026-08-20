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
import { cn } from "@/lib/utils"
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

// Compact token count: 812, 1.2k, 48k.
function formatTokens(n: number): string {
  if (n < 1000) return String(n)
  const k = n / 1000
  return `${k >= 10 ? Math.round(k) : k.toFixed(1)}k`
}

export function ConversationItem({
  chat,
  index,
}: {
  chat: ChatSummary
  index: number
}) {
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
    <li className="group relative border-b border-border before:absolute before:top-1/2 before:left-0 before:h-0 before:w-0.5 before:-translate-y-1/2 before:rounded-full before:bg-primary before:transition-all before:content-[''] hover:before:h-9">
      <div className="flex items-baseline gap-5 py-5 pr-1 pl-4">
        {/* Editorial index numeral — a table-of-contents cue. */}
        <span className="w-7 shrink-0 font-display text-xs text-muted-foreground tabular-nums transition-colors group-hover:text-primary">
          {String(index + 1).padStart(2, "0")}
        </span>

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
            className="h-9 flex-1"
          />
        ) : (
          <Link
            href={`/chat/${chat.id}`}
            className="min-w-0 flex-1 rounded-(--radius-marker) outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <span className="block truncate font-display text-lg leading-snug tracking-tight underline-offset-4 group-hover:underline">
              {chat.title}
            </span>
            <span className="eyebrow mt-1.5 flex items-center gap-2.5 text-muted-foreground tabular-nums">
              <span>{formatDate(chat.createdAt)}</span>
              {chat.tokens > 0 && (
                <>
                  <span className="text-muted-foreground/30">·</span>
                  <span>{formatTokens(chat.tokens)} tokens</span>
                </>
              )}
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
                className={cn(
                  "shrink-0 self-center text-muted-foreground opacity-0 transition-opacity",
                  "group-hover:opacity-100 focus-visible:opacity-100 data-[popup-open]:opacity-100 data-open:opacity-100"
                )}
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
      </div>

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
    </li>
  )
}
