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
import { useProfile } from "@/hooks/use-profile"

export function NavProfile() {
  const { isAuthenticated } = useAuth()
  const { user, isLoading } = useProfile()

  if (!isAuthenticated) {
    return (
      <SidebarMenu>
        <SidebarMenuItem>
          <SidebarMenuButton tooltip="Log in" render={<Link href="/login" />}>
            <LogIn />
            <span>Log in</span>
          </SidebarMenuButton>
        </SidebarMenuItem>
      </SidebarMenu>
    )
  }

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        {/* Default size, not `lg`. The footer is three rows — Admin, Settings,
            Profile — and a 48px row under two 32px ones read as a different kind
            of control rather than the emphasis it was meant to be. */}
        <SidebarMenuButton tooltip="Profile" render={<Link href="/profile" />}>
          <User />
          {isLoading ? (
            <Skeleton className="h-4 w-24" />
          ) : (
            <span>{user?.username ?? "Username"}</span>
          )}
        </SidebarMenuButton>
      </SidebarMenuItem>
    </SidebarMenu>
  )
}
