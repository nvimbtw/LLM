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
import { RunningDot } from "@/components/ui/running-dot"
import { useJobsContext } from "@/components/jobs/jobs-provider"
import { useAuth } from "@/hooks/use-auth"
import { KIND_LABEL, isRunning, type Job } from "@/lib/jobs"
import { cn } from "@/lib/utils"

// Keep the sidebar list short — the rest are one click away on /jobs. Same limit
// and same reasoning as the recent-conversations list above it.
const LIMIT = 5

/**
 * What the GPU is working on, wherever you are.
 *
 * Deliberately *not* a second entry called "Active Jobs" sitting beside the "Jobs"
 * link — two rows that name the same destination read as two destinations. It is a
 * labelled group in its own right, exactly like Conversations, so the label
 * describes a list rather than competing with a nav item.
 *
 * The group has no collapse toggle, unlike Conversations. It only exists while
 * there is work, and a collapsed-but-active queue would hide the one thing it was
 * added to show — a disclosure triangle on something that disappears by itself is
 * a control for nothing.
 */
export function NavJobs() {
  const { isAdmin } = useAuth()
  const { running, queued } = useJobsContext()

  if (!isAdmin) return null

  // Running first, then the line behind it — the same order as the /jobs ledger,
  // so the sidebar is a short window onto that page rather than its own view.
  const active = [...(running ? [running] : []), ...queued]
  if (active.length === 0) return null

  const shown = active.slice(0, LIMIT)
  const overflow = active.length - shown.length

  return (
    <SidebarGroup className="group-data-[collapsible=icon]:hidden">
      <SidebarGroupLabel className="text-xs font-medium tracking-[0.16em] text-sidebar-foreground/75 uppercase">
        Active
        <span className="ml-auto text-sidebar-foreground/75 tabular-nums">
          {active.length}
        </span>
      </SidebarGroupLabel>
      <SidebarGroupContent>
        <SidebarMenu>
          {shown.map((job, index) => (
            // Places in line are numbered from the first *waiting* job, so the
            // queue reads 1, 2, 3 whether or not something is currently running.
            <JobItem
              key={job.id_job}
              job={job}
              position={running ? index : index + 1}
            />
          ))}
          {overflow > 0 && (
            <SidebarMenuItem>
              <SidebarMenuButton
                render={<Link href="/jobs" />}
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

function JobItem({ job, position }: { job: Job; position: number }) {
  const pathname = usePathname()
  const isActive = pathname === `/jobs/${job.id_job}`
  const live = isRunning(job)
  const percent = Math.round(Math.min(1, Math.max(0, job.progress)) * 100)

  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        isActive={isActive}
        render={<Link href={`/jobs/${job.id_job}`} />}
        className={cn(
          // Two-line "index entry", matching the conversation rows: label over a
          // muted subtitle, with a left marginalia rule that grows on hover and
          // turns accent when active.
          "relative h-auto items-start gap-2 py-1.5 pl-3.5",
          "before:absolute before:top-1/2 before:left-1 before:h-0 before:w-px before:-translate-y-1/2",
          "before:rounded-full before:bg-sidebar-foreground/30 before:transition-all before:content-['']",
          "hover:before:h-4",
          isActive &&
            "before:h-4 before:w-0.5 before:bg-primary hover:before:bg-primary"
        )}
      >
        {/* The gutter says which of the two states a row is in without a word:
            the running job animates in its kind's mode, the waiting ones carry
            their place in line. Fixed width so the labels stay aligned. */}
        <span className="flex w-4 shrink-0 items-center justify-center pt-0.5">
          {live ? (
            <RunningDot
              label={`${KIND_LABEL[job.kind]} running`}
              className="bg-sidebar-foreground"
            />
          ) : (
            <span className="text-xs leading-4 text-sidebar-foreground/75 tabular-nums">
              {position}
            </span>
          )}
        </span>

        <span className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span className="w-full truncate text-sm leading-tight">
            {job.label}
          </span>
          <span className="flex w-full items-center justify-between gap-2 text-xs font-medium tracking-[0.1em] text-muted-foreground uppercase tabular-nums">
            <span className="truncate">
              {KIND_LABEL[job.kind]}
              {live && job.stage ? ` · ${job.stage}` : ""}
              {!live ? " · queued" : ""}
            </span>
            {live && <span className="shrink-0">{percent}%</span>}
          </span>
        </span>
      </SidebarMenuButton>
    </SidebarMenuItem>
  )
}
