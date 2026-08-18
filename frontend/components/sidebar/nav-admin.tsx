"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Shield } from "lucide-react"

import {
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from "@/components/ui/sidebar"
import { RunningDot } from "@/components/ui/running-dot"
import { useJobsContext } from "@/components/jobs/jobs-provider"
import { useAuth } from "@/hooks/use-auth"
import { KIND_LABEL } from "@/lib/jobs"

/**
 * One row for everything behind the admin flag.
 *
 * It used to be a labelled group of six — Tokenize, Train, Benchmark, Jobs,
 * Throbber, Markup — which made the admin surface the largest thing in a sidebar
 * whose subject is chat, and made a fixture page look like a peer of
 * Conversations. The six now live on `/admin` as a directory with room to say what
 * each one is; this row is the door.
 *
 * In the footer beside Settings rather than in the content above, because that is
 * what it is: not somewhere you go to work, but a mode of the app you have access
 * to. The footer is where the app keeps the things that are true of *you* rather
 * than of what you are reading.
 */
export function NavAdmin() {
  const pathname = usePathname()
  const { isAdmin } = useAuth()

  if (!isAdmin) return null

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <SidebarMenuButton
          tooltip="Admin"
          isActive={pathname === "/admin"}
          render={<Link href="/admin" />}
          className="[&>svg]:opacity-70 data-active:[&>svg]:opacity-100"
        >
          <Shield />
          <span>Admin</span>
          <QueueTell />
        </SidebarMenuButton>
      </SidebarMenuItem>
    </SidebarMenu>
  )
}

/**
 * The GPU's state on the Admin row itself.
 *
 * Redundant with the Active group above it while the sidebar is expanded — and
 * that is exactly when it earns its keep, because that group is hidden when the
 * sidebar collapses to icons. Then this mark is the only thing left saying the
 * machine is busy.
 *
 * Which is why it has to survive the collapse without shoving the icon off
 * centre: expanded it is a trailing mark on the row, collapsed it lifts out of
 * the flow into the corner of the 32px square and reads as a badge on the glyph.
 */
function QueueTell() {
  const { running } = useJobsContext()
  if (!running) return null

  return (
    <RunningDot
      label={`${KIND_LABEL[running.kind]} running`}
      className="ml-auto bg-sidebar-foreground group-data-[collapsible=icon]:pointer-events-none group-data-[collapsible=icon]:absolute group-data-[collapsible=icon]:top-1 group-data-[collapsible=icon]:right-1 group-data-[collapsible=icon]:ml-0"
    />
  )
}
