"use client"

import * as React from "react"
import { useSessionToken } from "@/hooks/use-auth"
import { useChatArchive } from "@/components/chat/chats-provider"
import { useProjectMembership } from "@/hooks/use-project-membership"
import { useProjects } from "./projects-provider"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { FormError, LoadError } from "@/components/ui/error-state"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"

export interface AddConversationProps {
  projectId: string
  open: boolean
  onOpenChange: (open: boolean) => void
}

/** Presents existing account conversations that can be added or moved into the selected project. */
export function AddConversation({
  projectId,
  open,
  onOpenChange,
}: AddConversationProps) {
  const token = useSessionToken()
  return (
    <Dialog open={open && Boolean(token)} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Add conversation</DialogTitle>
          <DialogDescription>
            Choose a chat to move into this project.
          </DialogDescription>
        </DialogHeader>
        <ConversationPicker
          key={`${token}:${projectId}`}
          projectId={projectId}
          open={open}
          onOpenChange={onOpenChange}
        />
      </DialogContent>
    </Dialog>
  )
}

/** Filters the real archive and performs one membership write; failed archive loads remain retryable. */
function ConversationPicker({
  projectId,
  open,
  onOpenChange,
}: AddConversationProps) {
  const { conversations, isLoading, error, refresh } = useChatArchive(open)
  const { catalog } = useProjects()
  const [query, setQuery] = React.useState("")
  const membership = useProjectMembership(() => onOpenChange(false))
  const available = conversations.filter(
    (chat) =>
      chat.projectId !== projectId &&
      chat.title.toLocaleLowerCase().includes(query.toLocaleLowerCase())
  )
  return (
    <div className="space-y-4">
      <Input
        aria-label="Find a conversation"
        placeholder="Find a conversation…"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        disabled={membership.busy}
      />
      {isLoading ? (
        <p className="py-6 text-sm text-muted-foreground" role="status">
          Loading conversations…
        </p>
      ) : error ? (
        <LoadError title={error} onRetry={() => void refresh()} />
      ) : available.length === 0 ? (
        <p className="py-6 text-sm text-muted-foreground">
          {query
            ? "No matching conversations."
            : "No conversations available to add."}
        </p>
      ) : (
        <ul className="max-h-80 space-y-1 overflow-y-auto">
          {available.map((chat) => (
            <li key={chat.id}>
              <Button
                variant="ghost"
                className="h-auto w-full justify-start py-3 text-left"
                disabled={membership.busy}
                onClick={() => void membership.move(chat.id, projectId)}
              >
                <span className="min-w-0">
                  <span className="block truncate">{chat.title}</span>
                  {chat.projectId && (
                    <span className="mt-1 block text-xs text-muted-foreground">
                      Move from{" "}
                      {catalog.projects.find(
                        (project) => project.project_id === chat.projectId
                      )?.name ?? "another project"}
                    </span>
                  )}
                </span>
              </Button>
            </li>
          ))}
        </ul>
      )}
      {membership.error && <FormError>{membership.error}</FormError>}
    </div>
  )
}
