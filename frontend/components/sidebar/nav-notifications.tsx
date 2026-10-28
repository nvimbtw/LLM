"use client"

import { Bell } from "lucide-react"
import { useNotifications } from "@/components/ui/toast"
import {
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from "@/components/ui/sidebar"

/** Opens account notification history and counts retained undismissed entries, independent of whether they were viewed. */
export function NavNotifications() {
  const { items, setOpen } = useNotifications()
  const count = items.filter((item) => item.lifetime !== "transient").length
  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <SidebarMenuButton
          tooltip="Notifications · Ctrl+Alt+N"
          aria-label={`Notifications, ${count} retained`}
          onClick={() => setOpen(true)}
        >
          <Bell />
          <span className="flex-1 group-data-[collapsible=icon]:hidden">
            Notifications
          </span>
          {count > 0 && (
            <span className="text-xs tabular-nums group-data-[collapsible=icon]:hidden">
              {count}
            </span>
          )}
        </SidebarMenuButton>
      </SidebarMenuItem>
    </SidebarMenu>
  )
}
