"use client"

import * as React from "react"
import Link from "next/link"
import { Folder, Plus } from "lucide-react"
import { useProjects } from "./projects-provider"
import { ProjectEditor } from "./project-editor"
import { Button } from "@/components/ui/button"
import { PageHeader } from "@/components/ui/page-header"
import { LoadError } from "@/components/ui/error-state"
import { Skeleton } from "@/components/ui/skeleton"

/** Renders the private project directory, bounded pagination and the create-project flow. */
export function ProjectsView() {
  const { catalog, refresh, loadMore } = useProjects()
  const [creating, setCreating] = React.useState(false)
  return (
    <>
      <PageHeader
        title="Projects"
        action={
          <Button onClick={() => setCreating(true)}>
            <Plus />
            New project
          </Button>
        }
      />
      {catalog.loading && catalog.projects.length === 0 ? (
        <div className="space-y-6" aria-label="Loading projects">
          {[0, 1, 2].map((index) => (
            <Skeleton key={index} className="h-20 w-full" />
          ))}
        </div>
      ) : catalog.error ? (
        <LoadError title={catalog.error} onRetry={() => void refresh()} />
      ) : catalog.projects.length === 0 ? (
        <div className="flex flex-col items-center gap-4 py-20 text-center">
          <Folder className="size-7 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">
            Keep related conversations, notes and instructions together.
          </p>
          <Button variant="outline" onClick={() => setCreating(true)}>
            Create your first project
          </Button>
        </div>
      ) : (
        <ul>
          {catalog.projects.map((project) => (
            <li key={project.project_id} className="border-b border-border">
              <Link
                href={`/projects/${project.project_id}`}
                className="flex items-start gap-4 rounded-(--radius-control) px-2 py-6 outline-none hover:bg-accent/40 focus-visible:ring-2 focus-visible:ring-ring"
              >
                <Folder className="mt-1 size-5 shrink-0 text-muted-foreground" />
                <div className="min-w-0 flex-1">
                  <h2 className="truncate font-display text-xl font-semibold">
                    {project.name}
                  </h2>
                  {project.description && (
                    <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">
                      {project.description}
                    </p>
                  )}
                </div>
                <span className="shrink-0 text-xs text-muted-foreground tabular-nums">
                  {project.thread_count}{" "}
                  {project.thread_count === 1 ? "chat" : "chats"}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
      {catalog.nextOffset < catalog.total && (
        <Button
          variant="outline"
          className="mt-6"
          onClick={() => void loadMore()}
          disabled={catalog.loading}
        >
          {catalog.loading ? "Loading…" : "Load more projects"}
        </Button>
      )}
      <ProjectEditor open={creating} onOpenChange={setCreating} />
    </>
  )
}
