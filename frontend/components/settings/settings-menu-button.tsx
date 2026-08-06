"use client"

import { Settings } from "lucide-react"

import {
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from "@/components/ui/sidebar"
import { useSettings } from "./settings-provider"

/** Sidebar entry point for the command modal — the discoverable twin of Ctrl+K. */
export function SidebarSettingsButton() {
  const { setCommandOpen } = useSettings()
  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <SidebarMenuButton
          tooltip="Settings"
          onClick={() => setCommandOpen(true)}
        >
          <Settings />
          <span className="group-data-[collapsible=icon]:hidden">Settings</span>
          <kbd className="ml-auto rounded border border-border bg-muted px-1.5 py-0.5 font-mono text-xs text-muted-foreground group-data-[collapsible=icon]:hidden">
            ⌘K
          </kbd>
        </SidebarMenuButton>
      </SidebarMenuItem>
    </SidebarMenu>
  )
}
