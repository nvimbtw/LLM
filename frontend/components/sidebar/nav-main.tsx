import Link from "next/link"
import {
  Boxes,
  ChartColumn,
  MessagesSquare,
  MessageSquarePlus,
} from "lucide-react"
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
  // Always visible (expanded and collapsed) — the entry point to the full
  // archive. The recent-chats list (NavConversations) sits below it when expanded.
  { title: "Conversations", href: "/conversations", icon: MessagesSquare },
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
