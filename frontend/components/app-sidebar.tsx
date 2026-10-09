import Link from "next/link"
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
import { NavMain } from "./sidebar/nav-main"
import { NavConversations } from "./sidebar/nav-conversations"
import { NavAdmin } from "./sidebar/nav-admin"
import { NavJobs } from "./sidebar/nav-jobs"
import { NavProfile } from "./sidebar/nav-profile"

// Brand shown in the sidebar header. Change these to rename the app:
// APP_NAME is shown when expanded, APP_NAME_SHORT when collapsed to icons.
const APP_NAME = "Slate"
const APP_NAME_SHORT = "S"

export function AppSidebar() {
  return (
    <Sidebar collapsible="icon">
      <SidebarHeader className="border-b border-sidebar-border/60 px-3 pt-4 pb-3 group-data-[collapsible=icon]:px-2">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              tooltip={APP_NAME}
              render={<Link href="/" />}
              className="h-auto py-0 hover:bg-transparent active:bg-transparent"
            >
              <span className="flex flex-col group-data-[collapsible=icon]:hidden">
                <span className="font-display text-xl leading-none font-semibold tracking-tight">
                  {APP_NAME}
                </span>
                <span className="mt-1 eyebrow text-sidebar-foreground/75">
                  LLM Project
                </span>
              </span>
              <span className="hidden font-display text-lg font-semibold tracking-tight group-data-[collapsible=icon]:inline">
                {APP_NAME_SHORT}
              </span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      {/* Collapsed, the rail is exactly 3rem and each row is a 2rem button, so the
          only padding that may exist is the 0.5rem either side that its own
          SidebarGroup already supplies. This `px-1` stacks on top of that and was
          the actual reason the glyphs sat left of centre — 12px in, 4px out. */}
      <SidebarContent className="gap-0.5 px-1 py-2 group-data-[collapsible=icon]:px-0">
        <NavMain />
        {/* What the GPU is doing, wherever you are — a window onto /admin, not a
            way in (that row is in the footer). Renders nothing at all when
            nothing is running or waiting. */}
        <NavJobs />
        <NavConversations />
      </SidebarContent>
      {/* The footer's rows are bare SidebarMenus with no group around them, so
          here the 0.5rem has to come from the footer itself — px-0 would push
          them hard against the edge. Different number, same 8 + 32 + 8 sum. */}
      <SidebarFooter className="border-t border-sidebar-border/60 px-1 py-2 group-data-[collapsible=icon]:px-2">
        <NavAdmin />
        <NavProfile />
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
