"use client"

import type { ReactNode } from "react"
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
import { PageHeader } from "@/components/ui/page-header"
import { FormError } from "@/components/ui/error-state"

export interface ProjectEditorProps {
  project?: Project
  actions?: ReactNode
  onSaved: (projectId: string) => void
}

/** Renders page-native project fields, resetting private drafts when account or project identity changes. */
export function ProjectEditor(props: ProjectEditorProps) {
  const token = useSessionToken()
  return token ? (
    <ProjectEditorForm
      key={token + ":" + (props.project?.project_id ?? "new")}
      {...props}
    />
  ) : null
}

/** Edits project metadata as heading and prose fields. Explicit saves preserve failed drafts and enforce Unicode budgets. */
function ProjectEditorForm({ project, actions, onSaved }: ProjectEditorProps) {
  const editor = useProjectEditor(project, onSaved)
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault()
        void editor.submit()
      }}
    >
      <PageHeader
        back={{ href: "/projects", label: "Projects" }}
        contentClassName="min-w-48 flex-1"
        title={
          <>
            <span className="sr-only">{project?.name ?? "New project"}</span>
            <Input
              aria-label="Project name"
              autoFocus={!project}
              required
              placeholder="Project name"
              maxLength={PROJECT_NAME_LIMIT * 2}
              value={editor.fields.name}
              onChange={(event) => editor.change("name", event.target.value)}
              disabled={editor.submitting}
              className="h-auto rounded-none border-0 bg-transparent px-0 py-0 font-display text-3xl font-semibold tracking-tight shadow-none focus-visible:ring-0 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring/50 md:text-3xl dark:bg-transparent"
            />
          </>
        }
        description={
          <>
            <Textarea
              aria-label="Project description"
              rows={1}
              placeholder="Add a description…"
              maxLength={PROJECT_DESCRIPTION_LIMIT * 2}
              value={editor.fields.description}
              onChange={(event) =>
                editor.change("description", event.target.value)
              }
              disabled={editor.submitting}
              className="min-h-5 resize-none rounded-none border-0 bg-transparent px-0 py-0 text-sm leading-relaxed text-muted-foreground shadow-none focus-visible:ring-0 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring/50 dark:bg-transparent"
            />
            <p className="text-xs text-muted-foreground">
              Private project
              {project && (
                <>
                  {" "}
                  · {project.thread_count}{" "}
                  {project.thread_count === 1
                    ? "conversation"
                    : "conversations"}
                </>
              )}
            </p>
          </>
        }
        action={
          <div className="flex flex-wrap items-center gap-2">
            {(!project || editor.hasChanges) && (
              <Button
                type="submit"
                variant={project ? "outline" : "default"}
                disabled={editor.submitting || !editor.fields.name.trim()}
              >
                {editor.submitting
                  ? "Saving…"
                  : project
                    ? "Save changes"
                    : "Create project"}
              </Button>
            )}
            {actions}
          </div>
        }
      />
      <section className="mb-10" aria-labelledby="project-notes-heading">
        <h2
          id="project-notes-heading"
          className="mb-3 font-display text-xl font-semibold"
        >
          Notes and instructions
        </h2>
        <Textarea
          aria-labelledby="project-notes-heading"
          rows={1}
          placeholder="Add notes and instructions…"
          maxLength={PROJECT_NOTES_LIMIT * 2}
          value={editor.fields.notes}
          onChange={(event) => editor.change("notes", event.target.value)}
          disabled={editor.submitting}
          className="min-h-6 resize-none rounded-none border-0 bg-transparent px-0 py-0 text-sm leading-relaxed shadow-none focus-visible:ring-0 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring/50 dark:bg-transparent"
        />
      </section>
      {editor.error && <FormError className="mb-6">{editor.error}</FormError>}
    </form>
  )
}
