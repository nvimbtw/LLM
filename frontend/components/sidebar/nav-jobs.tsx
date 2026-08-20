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
import { useAuth } from "@/hooks/use-auth"
import { KIND_LABEL, isRunning, jobHref, type Job } from "@/lib/jobs"
import { cn } from "@/lib/utils"

// Keep the sidebar list short — the rest are one click away on /admin. Same limit
// and same reasoning as the recent-conversations list above it.
const LIMIT = 5

/**
 * What the GPU is working on, wherever you are.
 *
 * Deliberately *not* a second entry called "Active Jobs" sitting beside the Admin
 * row — two rows that name the same destination read as two destinations. It is a
 * labelled group in its own right, exactly like Conversations, so the label
 * describes a list rather than competing with a nav item. That the Admin row now
 * lives in the footer helps: the group and the door are no longer adjacent.
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

  // Running first, then the line behind it — the same order as the /admin ledger,
  // so the sidebar is a short window onto that page rather than its own view.
  const active = [...(running ? [running] : []), ...queued]
  if (active.length === 0) return null

  const shown = active.slice(0, LIMIT)
  const overflow = active.length - shown.length

  return (
    <SidebarGroup className="group-data-[collapsible=icon]:hidden">
      <SidebarGroupLabel className="eyebrow text-sidebar-foreground/75">
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

function JobItem({ job, position }: { job: Job; position: number }) {
  const pathname = usePathname()
  const href = jobHref(job)
  const isActive = pathname === href
  const live = isRunning(job)
  const percent = Math.round(Math.min(1, Math.max(0, job.progress)) * 100)

  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        isActive={isActive}
        render={<Link href={href} />}
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
        <span className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span className="w-full truncate text-sm leading-tight">
            {job.label}
          </span>
          <span className="eyebrow flex w-full items-center justify-between gap-2 text-muted-foreground tabular-nums">
            <span className="truncate">
              {KIND_LABEL[job.kind]}
              {live && job.stage ? ` · ${job.stage}` : ""}
              {!live ? " · queued" : ""}
            </span>
            {/* One right-hand column, one number: how far along the running job
                is, or how far down the line a waiting one sits. */}
            <span className="shrink-0">
              {live ? `${percent}%` : `#${position}`}
            </span>
          </span>
        </span>
      </SidebarMenuButton>
    </SidebarMenuItem>
  )
}
