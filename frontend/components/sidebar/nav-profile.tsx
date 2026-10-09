"use client"

import Link from "next/link"
import { LogIn, User } from "lucide-react"
import {
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from "@/components/ui/sidebar"
import { Skeleton } from "@/components/ui/skeleton"
import { useAuth } from "@/hooks/use-auth"
import { useSettings } from "@/components/settings/settings-provider"
import { useProfile } from "@/hooks/use-profile"

export function NavProfile() {
  const { isAuthenticated } = useAuth()
  const { openSettings } = useSettings()
  const { user, isLoading } = useProfile()

  if (!isAuthenticated) {
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
