"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"

import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { useJobsContext } from "@/components/jobs/jobs-provider"
import { useAuth, useSessionToken } from "@/hooks/use-auth"
import { useTrainingDraft } from "@/hooks/use-training-draft"
import { SidebarEntryButton } from "./sidebar-entry-button"
import { KIND_LABEL, isRunning, jobHref, type Job } from "@/lib/jobs"

// Keep the sidebar list short — the rest are one click away on /admin. Same limit
// and same reasoning as the recent-conversations list above it.
const LIMIT = 5

/** Shows the account's local training draft before running/queued work. The expanded group remains visible while either exists. */
export function NavJobs() {
  const { isAdmin } = useAuth()
  const { running, queued } = useJobsContext()
  const token = useSessionToken()
  const { draft } = useTrainingDraft(isAdmin ? token : null)
  const pathname = usePathname()

  if (!isAdmin) return null

  const active = [...(running ? [running] : []), ...queued]
  if (active.length === 0 && !draft) return null

  const shown = active.slice(0, LIMIT)
  const overflow = active.length - shown.length

  return (
    <SidebarGroup className="group-data-[collapsible=icon]:hidden">
      <SidebarGroupLabel className="eyebrow text-sidebar-foreground/75">
        Currently working
        <span className="ml-auto text-sidebar-foreground/75 tabular-nums">
          {active.length + (draft ? 1 : 0)}
        </span>
      </SidebarGroupLabel>
      <SidebarGroupContent>
        <SidebarMenu>
          {draft && (
            <SidebarMenuItem>
              <SidebarEntryButton
                isActive={pathname === "/train/new-train"}
                render={<Link href="/train/new-train" />}
                className="h-auto items-start py-1.5"
              >
                <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                  <span className="w-full truncate text-sm leading-tight">
                    {draft.config.model_name.trim() || "Untitled training"}
                  </span>
                  <span className="eyebrow text-muted-foreground">
                    Train · Draft
                  </span>
                </span>
              </SidebarEntryButton>
            </SidebarMenuItem>
          )}
          {shown.map((job, index) => (
            <JobItem
              key={job.id_job}
              job={job}
              position={running ? index : index + 1}
            />
          ))}
          {overflow > 0 && (
            <SidebarMenuItem>
              <SidebarMenuButton
                render={<Link href="/admin" />}
                className="h-7 pl-3.5 text-xs text-sidebar-foreground/75"
              >
                {overflow} more in the queue
              </SidebarMenuButton>
            </SidebarMenuItem>
          )}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  )
}

/** Renders one active execution with progress or its waiting position, independently of the pinned draft. */
function JobItem({ job, position }: { job: Job; position: number }) {
  const pathname = usePathname()
  const href = jobHref(job)
  const isActive = pathname === href
  const live = isRunning(job)
  const percent = Math.round(Math.min(1, Math.max(0, job.progress)) * 100)

  return (
    <SidebarMenuItem>
      <SidebarEntryButton
        isActive={isActive}
        render={<Link href={href} />}
        className="h-auto items-start gap-2 py-1.5"
      >
        <span className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span className="w-full truncate text-sm leading-tight">
            {job.label}
          </span>
          <span className="flex w-full items-center justify-between gap-2 eyebrow text-muted-foreground tabular-nums">
            <span className="truncate">
              {KIND_LABEL[job.kind]}
              {live && job.stage ? ` · ${job.stage}` : ""}
              {!live ? " · queued" : ""}
            </span>
            <span className="shrink-0">
              {live ? `${percent}%` : `#${position}`}
            </span>
          </span>
        </span>
      </SidebarEntryButton>
    </SidebarMenuItem>
  )
}
