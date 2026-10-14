"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { MessageSquarePlus, Plus, Pencil, Trash2 } from "lucide-react"
import { useProject } from "@/hooks/use-project"
import { useProjectRemoval } from "@/hooks/use-project-removal"
import { PROJECT_PAGE_SIZE } from "@/lib/projects"
import { useProjects } from "./projects-provider"
import { ProjectEditor } from "./project-editor"
import { AddConversation } from "./add-conversation"
import { NewChat } from "@/components/chat/new-chat"
import { ConversationItem } from "@/components/conversations/conversation-item"
import { Button } from "@/components/ui/button"
import { PageHeader } from "@/components/ui/page-header"
import { Skeleton } from "@/components/ui/skeleton"
import { FormError, LoadError } from "@/components/ui/error-state"
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

export interface ProjectViewProps {
  projectId: string
}

/** Renders one private project's notes, existing conversations, real new-chat composer and project lifecycle controls. */
export function ProjectView({ projectId }: ProjectViewProps) {
  const view = useProject(projectId)
  const projects = useProjects()
  const router = useRouter()
  const [editing, setEditing] = React.useState(false)
  const [adding, setAdding] = React.useState(false)
  const [composing, setComposing] = React.useState(false)
  const [deleting, setDeleting] = React.useState(false)
  const removal = useProjectRemoval(projectId, () => router.push("/projects"))
  const project = view.project

  if (!project)
    return (
      <>
        <PageHeader
          title="Project"
          back={{ href: "/projects", label: "Projects" }}
        />
        {view.isLoading ? (
          <div className="space-y-6" aria-label="Loading project">
            <Skeleton className="h-8 w-48" />
            <Skeleton className="h-32 w-full" />
          </div>
        ) : (
          <LoadError
            title={view.error ?? "This project is no longer available."}
            onRetry={view.refresh}
          />
        )}
      </>
    )

  return (
    <>
      <PageHeader
        title={project.name}
        description={project.description || undefined}
        back={{ href: "/projects", label: "Projects" }}
        action={
          <Button onClick={() => setComposing(true)}>
            <MessageSquarePlus />
            New chat
          </Button>
        }
      />
      <div className="mb-8 flex flex-wrap items-center justify-between gap-3">
        <span className="text-xs text-muted-foreground">
          Private project · {project.threads_total}{" "}
          {project.threads_total === 1 ? "conversation" : "conversations"}
        </span>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="sm" onClick={() => setEditing(true)}>
            <Pencil />
            Edit project
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setDeleting(true)}>
            <Trash2 />
            Delete project
          </Button>
        </div>
      </div>
      <section className="mb-10" aria-labelledby="project-notes-heading">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2
            id="project-notes-heading"
            className="font-display text-xl font-semibold"
          >
            Notes and instructions
          </h2>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setEditing(true)}
            aria-label="Edit notes and instructions"
          >
            Edit
          </Button>
        </div>
        {project.notes ? (
          <p className="rounded-(--radius-sheet) border border-border px-5 py-4 text-sm leading-relaxed break-words whitespace-pre-wrap">
            {project.notes}
          </p>
        ) : (
          <p className="text-sm text-muted-foreground">
            Add context and instructions to use across this project’s
            conversations.
          </p>
        )}
      </section>
      {composing && (
        <section className="mb-10" aria-labelledby="project-new-chat-heading">
          <div className="mb-3 flex items-center justify-between">
            <h2
              id="project-new-chat-heading"
              className="font-display text-xl font-semibold"
            >
              New chat
            </h2>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setComposing(false)}
            >
              Close draft
            </Button>
          </div>
          <NewChat projectId={projectId} embedded />
        </section>
      )}
      <section aria-labelledby="project-conversations-heading">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-4">
          <h2
            id="project-conversations-heading"
            className="font-display text-xl font-semibold"
          >
            Conversations
          </h2>
          <Button variant="outline" size="sm" onClick={() => setAdding(true)}>
            <Plus />
            Add conversation
          </Button>
        </div>
        {view.error ? (
          <LoadError
            className="mt-4"
            title={view.error}
            onRetry={view.refresh}
          />
        ) : project.threads.length === 0 ? (
          <p className="py-12 text-center text-sm text-muted-foreground">
            Start a new chat or add an existing conversation.
          </p>
        ) : (
          <ol aria-busy={view.isLoading}>
            {project.threads.map((chat, index) => (
              <ConversationItem
                key={chat.id}
                chat={chat}
                index={view.offset + index}
                onChanged={() => {
                  view.refresh()
                  void projects.refresh()
                }}
              />
            ))}
          </ol>
        )}
        {project.threads_total > PROJECT_PAGE_SIZE && (
          <div className="mt-6 flex items-center justify-between gap-3">
            <Button
              variant="outline"
              size="sm"
              disabled={view.offset === 0 || view.isLoading}
              onClick={() =>
                view.changePage(Math.max(0, view.offset - PROJECT_PAGE_SIZE))
              }
            >
              Previous
            </Button>
            <span className="text-xs text-muted-foreground">
              {view.offset + 1}–
              {Math.min(
                view.offset + project.threads.length,
                project.threads_total
              )}{" "}
              of {project.threads_total}
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={
                view.offset + PROJECT_PAGE_SIZE >= project.threads_total ||
                view.isLoading
              }
              onClick={() => view.changePage(view.offset + PROJECT_PAGE_SIZE)}
            >
              Next
            </Button>
          </div>
        )}
      </section>
      <ProjectEditor
        open={editing}
        onOpenChange={setEditing}
        project={project}
      />
      <AddConversation
        projectId={projectId}
        open={adding}
        onOpenChange={setAdding}
      />
      <AlertDialog open={deleting} onOpenChange={setDeleting}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this project?</AlertDialogTitle>
            <AlertDialogDescription>
              {project.name} will be removed. Its conversations and messages
              will stay in your archive.
            </AlertDialogDescription>
          </AlertDialogHeader>
          {removal.error && <FormError>{removal.error}</FormError>}
          <AlertDialogFooter>
            <AlertDialogCancel disabled={removal.busy}>
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={removal.busy}
              onClick={() => void removal.confirm()}
            >
              {removal.busy ? "Deleting…" : "Delete project"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
