"use client"

import Link from "next/link"
import { LogIn, User } from "lucide-react"
import {
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from "@/components/ui/sidebar"
import { Skeleton } from "@/components/ui/skeleton"
import { useSessionToken } from "@/hooks/use-auth"
import { useSettings } from "@/components/settings/settings-provider"
import { useProfile } from "@/hooks/use-profile"

/** Shows anonymous navigation without profile requests and remounts signed-in profile reads for each actual session. */
export function NavProfile() {
  const token = useSessionToken()
  if (token) return <SignedInProfile key={token} />

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <SidebarMenuButton tooltip="Log in" render={<Link href="/login" />}>
          <LogIn />
          <span className="group-data-[collapsible=icon]:hidden">Log in</span>
        </SidebarMenuButton>
      </SidebarMenuItem>
    </SidebarMenu>
  )
}

/** Loads one mounted account's profile and opens its settings from the sidebar. */
function SignedInProfile() {
  const { openSettings } = useSettings()
  const { user, isLoading } = useProfile()

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <SidebarMenuButton
          tooltip="Settings and profile"
          aria-label="Settings and profile"
          onClick={() => openSettings("profile")}
        >
          <User />
          {isLoading ? (
            <Skeleton className="h-4 w-24 group-data-[collapsible=icon]:hidden" />
          ) : (
            <span className="min-w-0 flex-1 truncate group-data-[collapsible=icon]:hidden">
              {user?.username ?? "Username"}
            </span>
          )}
          <kbd className="ml-auto font-mono text-xs text-sidebar-foreground/60 group-data-[collapsible=icon]:hidden">
            ⌘K
          </kbd>
        </SidebarMenuButton>
      </SidebarMenuItem>
    </SidebarMenu>
  )
}
