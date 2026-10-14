import Link from "next/link"
import { Boxes, MessagesSquare, MessageSquarePlus, Folder } from "lucide-react"
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
  { title: "Conversations", href: "/conversations", icon: MessagesSquare },
  { title: "Projects", href: "/projects", icon: Folder },
]

/** Renders primary navigation, keeping Projects directly underneath the full conversation archive. */
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
