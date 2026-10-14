"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { useProjects } from "@/components/projects/projects-provider"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { SidebarEntryButton } from "./sidebar-entry-button"

/** Renders flat project links above recent chats, matching conversation rows and retaining a retryable catalogue failure. */
export function NavProjects() {
  const pathname = usePathname()
  const { catalog, refresh } = useProjects()
  return (
    <SidebarGroup className="group-data-[collapsible=icon]:hidden">
      <SidebarGroupLabel className="eyebrow text-sidebar-foreground/75">
        Projects
      </SidebarGroupLabel>
      <SidebarGroupContent>
        <SidebarMenu>
          {catalog.loading && catalog.projects.length === 0 ? (
            [0, 1].map((index) => (
              <SidebarMenuItem key={index}>
                <Skeleton className="mx-2 my-2 h-4 w-24" />
              </SidebarMenuItem>
            ))
          ) : catalog.error ? (
            <SidebarMenuItem>
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start text-xs"
                onClick={() => void refresh()}
              >
                Retry loading projects
              </Button>
            </SidebarMenuItem>
          ) : catalog.projects.length === 0 ? (
            <SidebarMenuItem>
              <SidebarMenuButton
                render={<Link href="/projects" />}
                className="text-xs text-sidebar-foreground/60"
              >
                Create a project
              </SidebarMenuButton>
            </SidebarMenuItem>
          ) : (
            catalog.projects.slice(0, 8).map((project) => (
              <SidebarMenuItem key={project.project_id}>
                <SidebarEntryButton
                  isActive={pathname === `/projects/${project.project_id}`}
                  render={<Link href={`/projects/${project.project_id}`} />}
                >
                  <span className="min-w-0 flex-1 truncate">{project.name}</span>
                </SidebarEntryButton>
              </SidebarMenuItem>
            ))
          )}
          {catalog.total > 8 && (
            <SidebarMenuItem>
              <SidebarMenuButton
                render={<Link href="/projects" />}
                className="text-xs text-sidebar-foreground/60"
              >
                All projects
              </SidebarMenuButton>
            </SidebarMenuItem>
          )}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  )
}
