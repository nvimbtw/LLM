"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Shield } from "lucide-react"

import {
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from "@/components/ui/sidebar"
import { useAuth } from "@/hooks/use-auth"

/**
 * One row for everything behind the admin flag.
 *
 * It used to be a labelled group of six — Tokenize, Train, Benchmark, Jobs,
 * Throbber, Markup — which made the admin surface the largest thing in a sidebar
 * whose subject is chat, and made a fixture page look like a peer of
 * Conversations. The six now live on `/admin` as a directory with room to say what
 * each one is; this row is the door.
 *
 * In the footer above the profile/settings entry rather than in the content above, because that is
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
        </SidebarMenuButton>
      </SidebarMenuItem>
    </SidebarMenu>
  )
}
