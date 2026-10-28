"use client"

import { EmptyState } from "@/components/ui/empty-state"

import Link from "next/link"
import { Folder, Plus } from "lucide-react"
import { useProjects } from "./projects-provider"
import { Button } from "@/components/ui/button"
import { PageHeader } from "@/components/ui/page-header"
import { LoadError } from "@/components/ui/error-state"
import {
  ResourceList,
  ResourceRow,
  ResourceLink,
  ResourceTitle,
} from "@/components/ui/resource-list"
import { Skeleton } from "@/components/ui/skeleton"

/** Renders the private project directory, bounded pagination and the create-project flow. */
export function ProjectsView() {
  const { catalog, refresh, loadMore } = useProjects()
  return (
    <>
      <PageHeader
        title="Projects"
        action={
          <Button
            nativeButton={false}
            render={<Link href="/projects/new-project" />}
          >
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
        <EmptyState
          title="No projects yet"
          action={
            <Button
              variant="outline"
              nativeButton={false}
              render={<Link href="/projects/new-project" />}
            >
              Create your first project
            </Button>
          }
        >
          Keep related conversations, notes and instructions together.
        </EmptyState>
      ) : (
        <ResourceList>
          {catalog.projects.map((project) => (
            <ResourceRow key={project.project_id}>
              <ResourceLink
                href={`/projects/${project.project_id}`}
                className="flex items-start gap-4 rounded-(--radius-control) py-0 outline-none hover:bg-accent/40 focus-visible:ring-2 focus-visible:ring-ring"
              >
                <Folder className="mt-1 size-5 shrink-0 text-muted-foreground" />
                <div className="min-w-0 flex-1">
                  <ResourceTitle>{project.name}</ResourceTitle>
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
              </ResourceLink>
            </ResourceRow>
          ))}
        </ResourceList>
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
    </>
  )
}
