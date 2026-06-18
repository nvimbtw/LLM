import Link from "next/link";
import { Boxes, ChartColumn, MessagesSquare, MessageSquarePlus } from "lucide-react";
import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from "@/components/ui/sidebar"

const items = [
  { title: "New", href: "/", icon: MessageSquarePlus },
  { title: "Models", href: "/models", icon: Boxes },
  { title: "Usage", href: "/usage", icon: ChartColumn },
]

export function NavMain() {
  return (
    <SidebarGroup>
      <SidebarGroupContent>
        <SidebarMenu>
          {items.map((item) => (
            <SidebarMenuItem key={item.href}>
              <SidebarMenuButton
                tooltip={item.title}
                render={<Link href={item.href} />}
              >
                <item.icon />
                <span>{item.title}</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
          ))}
          {/* Only shown when the sidebar is collapsed to icons; the full
              conversation list (NavConversations) is hidden in that state. */}
          <SidebarMenuItem className="hidden group-data-[collapsible=icon]:block">
            <SidebarMenuButton
              tooltip="Conversations"
              render={<Link href="/conversations" />}
            >
              <MessagesSquare />
              <span>Conversations</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  )
}
