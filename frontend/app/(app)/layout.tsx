import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"
import { MobileNavigation } from "@/components/sidebar/mobile-navigation"

/** Renders signed-in navigation and a bounded, independently scrolling content column. Desktop uses the collapsible icon rail; mobile keeps a compact visible opener for its sidebar sheet. */
export default function AppLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset className="flex h-svh min-w-0 flex-col overflow-hidden">
        <MobileNavigation />
        <main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
          {children}
        </main>
      </SidebarInset>
    </SidebarProvider>
  )
}
