"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Gauge, ListOrdered, Repeat, Sparkles } from "lucide-react"

import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from "@/components/ui/sidebar"
import { Throbber } from "@/components/ui/throbber"
import { useJobsContext } from "@/components/jobs/jobs-provider"
import { useAuth } from "@/hooks/use-auth"
import { throbberMode } from "@/lib/jobs"

// Train is not here on purpose: it is a way of putting work on the GPU, and every
// one of those now starts from Jobs. Two doors to the same queue is how a
// benchmark used to be launchable into a running training job.
//
// Files and Throbber were dropped to keep this list to the things worth a
// permanent slot. Both pages still exist at /files and /throbber.
const items = [
  { title: "Benchmark", href: "/benchmark", icon: Gauge },
  { title: "Jobs", href: "/jobs", icon: ListOrdered },
  // Cycles sit above Jobs conceptually — one cycle is a dozen jobs — but below it
  // here, because the queue is what you check constantly and a cycle is what you
  // check twice a day.
  { title: "Cycles", href: "/cycles", icon: Repeat },
  // The corpus, not the composer: this page is what you check before deciding to
  // generate, and generation is queued from the job menu like every other kind.
  { title: "SFT corpus", href: "/synth", icon: Sparkles },
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
    <Throbber
      mode={throbberMode(running)}
      accent="var(--sidebar-foreground)"
      className="ml-auto h-3.5 group-data-[collapsible=icon]:pointer-events-none group-data-[collapsible=icon]:absolute group-data-[collapsible=icon]:top-0 group-data-[collapsible=icon]:right-0 group-data-[collapsible=icon]:ml-0 group-data-[collapsible=icon]:h-2"
    />
  )
}
