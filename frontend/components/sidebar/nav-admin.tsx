"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Gauge, ListOrdered, Type, Brain, Loader, MessageSquareCode } from "lucide-react"

import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from "@/components/ui/sidebar"
import { RunningDot } from "@/components/ui/running-dot"
import { useJobsContext } from "@/components/jobs/jobs-provider"
import { useAuth } from "@/hooks/use-auth"
import { KIND_LABEL } from "@/lib/jobs"

const items = [
  { title: "Tokenize", href: "/tokenizers", icon: Type },
  { title: "Train", href: "/jobs/new/train", icon: Brain },
  { title: "Benchmark", href: "/benchmark", icon: Gauge },
  { title: "Jobs", href: "/jobs", icon: ListOrdered },
  { title: "Throbber", href: "/throbber", icon: Loader },
  { title: "Markup", href: "/markup", icon: MessageSquareCode },
]

export function NavAdmin() {
  const pathname = usePathname()
  const { isAdmin } = useAuth()

  if (!isAdmin) return null

  return (
    <SidebarGroup>
      <SidebarGroupLabel className="text-xs font-medium tracking-[0.16em] text-sidebar-foreground/75 uppercase">
        Admin
      </SidebarGroupLabel>
      <SidebarGroupContent>
        <SidebarMenu>
          {items.map((item) => (
            <SidebarMenuItem key={item.href}>
              <SidebarMenuButton
                tooltip={item.title}
                isActive={pathname === item.href}
                render={<Link href={item.href} />}
                className="[&>svg]:opacity-70 data-active:[&>svg]:opacity-100"
              >
                <item.icon />
                <span>{item.title}</span>
                {item.href === "/jobs" && <QueueTell />}
              </SidebarMenuButton>
            </SidebarMenuItem>
          ))}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  )
}

/**
 * The GPU's state on the Jobs row itself.
 *
 * Redundant with the Active group below it while the sidebar is expanded — and
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
