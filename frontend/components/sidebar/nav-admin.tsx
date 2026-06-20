"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { LoaderCircle, FileUp } from "lucide-react"

import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from "@/components/ui/sidebar"
import { useAuth } from "@/hooks/use-auth"

const items = [
  { title: "Files", href: "/files", icon: FileUp },
  { title: "Throbber", href: "/throbber", icon: LoaderCircle },
]

export function NavAdmin() {
  const pathname = usePathname()
  const { isAdmin } = useAuth()

  if (!isAdmin) return null

  return (
    <SidebarGroup>
      <SidebarGroupLabel>Admin</SidebarGroupLabel>
      <SidebarGroupContent>
        <SidebarMenu>
          {items.map((item) => (
            <SidebarMenuItem key={item.href}>
              <SidebarMenuButton
                tooltip={item.title}
                isActive={pathname === item.href}
                render={<Link href={item.href} />}
              >
                <item.icon />
                <span>{item.title}</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
          ))}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  )
}
