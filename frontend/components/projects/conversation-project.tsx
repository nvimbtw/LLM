"use client"

import * as React from "react"
import { useSessionToken } from "@/hooks/use-auth"
import { useProjectMembership } from "@/hooks/use-project-membership"
import type { ChatSummary } from "@/lib/chats"
import { useProjects } from "./projects-provider"
import { Button } from "@/components/ui/button"
import { FormError, LoadError } from "@/components/ui/error-state"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"

const NO_PROJECT = "none"

export interface ConversationProjectDialogProps {
  chat: ChatSummary
  open: boolean
  onOpenChange: (open: boolean) => void
}

/** Presents project membership editing without altering the conversation or its message history. */
export function ConversationProjectDialog({
  chat,
  open,
  onOpenChange,
}: ConversationProjectDialogProps) {
  const token = useSessionToken()
  return (
    <Dialog open={open && Boolean(token)} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Conversation project</DialogTitle>
          <DialogDescription>{chat.title}</DialogDescription>
        </DialogHeader>
        <ConversationProjectForm
          key={`${token}:${chat.id}`}
          chat={chat}
          open={open}
          onOpenChange={onOpenChange}
        />
      </DialogContent>
    </Dialog>
  )
}

/** Renders a project picker and applies only its explicitly saved selection; failed loads cannot remove membership. */
function ConversationProjectForm({
  chat,
  onOpenChange,
}: ConversationProjectDialogProps) {
  const { catalog, refresh, loadMore } = useProjects()
  const [selected, setSelected] = React.useState(chat.projectId ?? NO_PROJECT)
  const membership = useProjectMembership(() => onOpenChange(false))
  const currentMissing =
    Boolean(chat.projectId) &&
    !catalog.projects.some((project) => project.project_id === chat.projectId)
  return (
    <form
      className="space-y-4"
      onSubmit={(event) => {
        event.preventDefault()
        void membership.move(chat.id, selected === NO_PROJECT ? null : selected)
      }}
    >
      {catalog.error ? (
        <LoadError title={catalog.error} onRetry={() => void refresh()} />
      ) : (
        <>
          <label className="block text-sm" htmlFor="conversation-project">
            Project
          </label>
          <Select
            value={selected}
            onValueChange={(value) => {
              if (value) setSelected(value)
            }}
            disabled={membership.busy || catalog.loading}
          >
            <SelectTrigger id="conversation-project" className="w-full">
              <SelectValue>
                {(value) =>
                  value === NO_PROJECT
                    ? "No project"
                    : (catalog.projects.find(
                        (project) => project.project_id === value
                      )?.name ?? "Current project")
                }
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={NO_PROJECT}>No project</SelectItem>
              {currentMissing && (
                <SelectItem value={chat.projectId!}>Current project</SelectItem>
              )}
              {catalog.projects.map((project) => (
                <SelectItem key={project.project_id} value={project.project_id}>
                  {project.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {catalog.nextOffset < catalog.total && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              disabled={catalog.loading}
              onClick={() => void loadMore()}
            >
              Load more projects
            </Button>
          )}
        </>
      )}
      {membership.error && <FormError>{membership.error}</FormError>}
      <DialogFooter>
        <Button
          type="button"
          variant="outline"
          onClick={() => onOpenChange(false)}
          disabled={membership.busy}
        >
          Cancel
        </Button>
        <Button
          type="submit"
          disabled={
            membership.busy ||
            catalog.loading ||
            Boolean(catalog.error) ||
            selected === (chat.projectId ?? NO_PROJECT)
          }
        >
          {membership.busy ? "Saving…" : "Save"}
        </Button>
      </DialogFooter>
    </form>
  )
}
