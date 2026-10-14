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
import { NavProjects } from "./sidebar/nav-projects"
import { NavConversations } from "./sidebar/nav-conversations"
import { NavAdmin } from "./sidebar/nav-admin"
import { NavJobs } from "./sidebar/nav-jobs"
import { NavProfile } from "./sidebar/nav-profile"

// Brand shown in the sidebar header. Change these to rename the app:
// APP_NAME is shown when expanded, APP_NAME_SHORT when collapsed to icons.
const APP_NAME = "Slate"
const APP_NAME_SHORT = "S"

/** Renders the application navigation with a flat project section above recent ordinary conversations. */
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
      <SidebarContent className="gap-0.5 px-1 py-2 group-data-[collapsible=icon]:px-0">
        <NavMain />
        <NavJobs />
        <NavProjects />
        <NavConversations />
      </SidebarContent>
      <SidebarFooter className="border-t border-sidebar-border/60 px-1 py-2 group-data-[collapsible=icon]:px-2">
        <NavAdmin />
        <NavProfile />
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
