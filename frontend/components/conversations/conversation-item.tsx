"use client"

import * as React from "react"
import Link from "next/link"
import { MoreHorizontal, Pencil, Trash2, FolderInput } from "lucide-react"

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { DeleteConfirmation } from "@/components/ui/delete-confirmation"
import { Button } from "@/components/ui/button"
import { ConversationRename } from "@/components/chat/conversation-rename"
import { ConversationProjectDialog } from "@/components/projects/conversation-project"
import { useProjects } from "@/components/projects/projects-provider"
import {
  ResourceRow,
  ResourceTitle,
  ResourceMetadata,
} from "@/components/ui/resource-list"
import { cn } from "@/lib/utils"
import { useChats, type ChatSummary } from "@/components/chat/chats-provider"

/** Formats a stored conversation date for the current locale, omitting invalid values. */
function formatDate(iso: string) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ""
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  })
}

/** Formats the cumulative token count compactly for archive comparisons. */
function formatTokens(n: number): string {
  if (n < 1000) return String(n)
  const k = n / 1000
  return `${k >= 10 ? Math.round(k) : k.toFixed(1)}k`
}

/** Renders an archive or project conversation with rename, deletion and membership actions. */
export function ConversationItem({
  chat,
  index,
  onChanged,
}: {
  chat: ChatSummary
  index: number
  onChanged?: () => void
}) {
  const { renameChat, deleteChat } = useChats()
  const { catalog, refresh: refreshProjects } = useProjects()
  const [projectOpen, setProjectOpen] = React.useState(false)
  const project = catalog.projects.find(
    (entry) => entry.project_id === chat.projectId
  )

  const [isRenaming, setRenaming] = React.useState(false)
  const [confirmOpen, setConfirmOpen] = React.useState(false)

  return (
    <ResourceRow>
      <div className="flex items-baseline gap-5 pr-1">
        <span className="w-7 shrink-0 font-display text-xs text-muted-foreground tabular-nums transition-colors group-hover:text-primary">
          {String(index + 1).padStart(2, "0")}
        </span>

        {isRenaming ? (
          <ConversationRename
            title={chat.title}
            onSave={async (next) => {
              await renameChat(chat.id, next)
              onChanged?.()
            }}
            onDone={() => setRenaming(false)}
          />
        ) : (
          <Link
            href={`/chat/${chat.id}`}
            className="min-w-0 flex-1 rounded-(--radius-marker) outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <ResourceTitle>{chat.title}</ResourceTitle>
            <ResourceMetadata className="tabular-nums">
              <span>{formatDate(chat.createdAt)}</span>
              {chat.projectId && (
                <span className="max-w-32 truncate">
                  {project?.name ?? "Project"}
                </span>
              )}
              {chat.tokens > 0 && (
                <>
                  <span className="text-muted-foreground/30">·</span>
                  <span>{formatTokens(chat.tokens)} tokens</span>
                </>
              )}
            </ResourceMetadata>
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
                  "size-10 shrink-0 self-center text-muted-foreground opacity-100 transition-opacity md:opacity-70 [@media(pointer:coarse)]:opacity-100",
                  "group-hover:opacity-100 focus-visible:opacity-100 data-[popup-open]:opacity-100 data-open:opacity-100"
                )}
              >
                <MoreHorizontal />
              </Button>
            }
          />
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => setProjectOpen(true)}>
              <FolderInput />
              {chat.projectId
                ? "Move or remove from project"
                : "Add to project"}
            </DropdownMenuItem>
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

      <ConversationProjectDialog
        chat={chat}
        open={projectOpen}
        onOpenChange={setProjectOpen}
      />
      {confirmOpen && (
        <DeleteConfirmation
          open
          onOpenChange={setConfirmOpen}
          title="Delete this conversation?"
          description={`“${chat.title}” and all its messages will be permanently deleted. This cannot be undone.`}
          label="Delete conversation"
          onDelete={() => deleteChat(chat.id)}
          onDeleted={() => {
            if (chat.projectId) void refreshProjects()
            onChanged?.()
          }}
        />
      )}
    </ResourceRow>
  )
}
