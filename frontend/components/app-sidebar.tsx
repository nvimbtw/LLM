import Link from "next/link";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarRail,
} from "@/components/ui/sidebar"
import { NavMain } from "./sidebar/nav-main";
import { NavConversations } from "./sidebar/nav-conversations";
import { NavAdmin } from "./sidebar/nav-admin";
import { NavProfile } from "./sidebar/nav-profile";
import { SidebarSettingsButton } from "./settings/settings-menu-button";

// Brand shown in the sidebar header. Change these to rename the app:
// APP_NAME is shown when expanded, APP_NAME_SHORT when collapsed to icons.
const APP_NAME = "Lattice"
const APP_NAME_SHORT = "L"

export function AppSidebar() {
  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              tooltip={APP_NAME}
              render={<Link href="/" />}
              className="font-display text-lg font-semibold tracking-tight"
            >
              <span className="group-data-[collapsible=icon]:hidden">
                {APP_NAME}
              </span>
              <span className="hidden group-data-[collapsible=icon]:inline">
                {APP_NAME_SHORT}
              </span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <NavMain />
        <NavAdmin />
        <NavConversations />
      </SidebarContent>
      <SidebarFooter>
        <SidebarSettingsButton />
        <NavProfile />
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
