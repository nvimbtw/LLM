"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { LoaderCircle, FileUp, Dumbbell, Gauge } from "lucide-react"

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
  { title: "Train", href: "/training", icon: Dumbbell },
  { title: "Benchmark", href: "/benchmark", icon: Gauge },
  { title: "Throbber", href: "/throbber", icon: LoaderCircle },
]

export function NavAdmin() {
  const pathname = usePathname()
  const { isAdmin } = useAuth()

  if (!isAdmin) return null

  return (
    <SidebarGroup>
      <SidebarGroupLabel className="text-[0.62rem] font-medium uppercase tracking-[0.16em] text-sidebar-foreground/50">
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
              </SidebarMenuButton>
            </SidebarMenuItem>
          ))}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  )
}
