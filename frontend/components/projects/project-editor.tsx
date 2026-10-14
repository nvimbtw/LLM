"use client"

import { useRouter } from "next/navigation"
import { useSessionToken } from "@/hooks/use-auth"
import { useProjectEditor } from "@/hooks/use-project-editor"
import {
  PROJECT_NAME_LIMIT,
  PROJECT_DESCRIPTION_LIMIT,
  PROJECT_NOTES_LIMIT,
  type Project,
} from "@/lib/projects"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { FormError } from "@/components/ui/error-state"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"

export interface ProjectEditorProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  project?: Project
}

/** Presents an accessible create/edit dialog, resetting private drafts when its project or account changes. */
export function ProjectEditor({
  open,
  onOpenChange,
  project,
}: ProjectEditorProps) {
  const token = useSessionToken()
  return (
    <Dialog open={open && Boolean(token)} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[calc(100svh-2rem)] max-w-2xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{project ? "Edit project" : "New project"}</DialogTitle>
          <DialogDescription>
            Only you can access this project and its conversations.
          </DialogDescription>
        </DialogHeader>
        <ProjectEditorForm
          key={`${token}:${project?.project_id ?? "new"}`}
          project={project}
          onOpenChange={onOpenChange}
        />
      </DialogContent>
    </Dialog>
  )
}

export interface ProjectEditorFormProps {
  project?: Project
  onOpenChange: (open: boolean) => void
}

/** Renders labelled project fields and submits through the session-owned editor hook. HTML limits allow two UTF-16 units per Unicode character; submission enforces the exact character budgets. */
function ProjectEditorForm({ project, onOpenChange }: ProjectEditorFormProps) {
  const router = useRouter()
  const editor = useProjectEditor(project, (projectId) => {
    onOpenChange(false)
    if (!project) router.push(`/projects/${projectId}`)
  })
  return (
    <form
      className="space-y-5"
      onSubmit={(event) => {
        event.preventDefault()
        void editor.submit()
      }}
    >
      <label className="block space-y-2 text-sm">
        <span>Name</span>
        <Input
          autoFocus
          required
          maxLength={PROJECT_NAME_LIMIT * 2}
          value={editor.fields.name}
          onChange={(event) => editor.change("name", event.target.value)}
          disabled={editor.submitting}
        />
      </label>
      <label className="block space-y-2 text-sm">
        <span>Description</span>
        <Textarea
          rows={2}
          maxLength={PROJECT_DESCRIPTION_LIMIT * 2}
          value={editor.fields.description}
          onChange={(event) => editor.change("description", event.target.value)}
          disabled={editor.submitting}
        />
      </label>
      <label className="block space-y-2 text-sm">
        <span>Notes and instructions</span>
        <Textarea
          rows={8}
          maxLength={PROJECT_NOTES_LIMIT * 2}
          value={editor.fields.notes}
          onChange={(event) => editor.change("notes", event.target.value)}
          disabled={editor.submitting}
          aria-describedby="project-notes-hint"
        />
        <span
          id="project-notes-hint"
          className="block text-xs text-muted-foreground"
        >
          Shared context for conversations in this project.
        </span>
      </label>
      {editor.error && <FormError>{editor.error}</FormError>}
      <DialogFooter>
        <Button
          type="button"
          variant="outline"
          onClick={() => onOpenChange(false)}
          disabled={editor.submitting}
        >
          Cancel
        </Button>
        <Button
          type="submit"
          disabled={editor.submitting || !editor.fields.name.trim()}
        >
          {editor.submitting
            ? "Saving…"
            : project
              ? "Save changes"
              : "Create project"}
        </Button>
      </DialogFooter>
    </form>
  )
}
