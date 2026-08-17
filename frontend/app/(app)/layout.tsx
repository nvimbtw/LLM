import {
  SidebarProvider,
  SidebarTrigger,
  SidebarInset,
} from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"

/**
 * The signed-in frame: sidebar, header, scroll container. Everything except
 * `/login` and `/signup` renders inside it.
 */
export default function AppLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <SidebarProvider>
      <AppSidebar />
      {/* h-svh + overflow-hidden bound the column so page content scrolls inside
          <main>, keeping the header (and the chat composer) pinned instead of
          scrolling the document body. */}
      {/* min-w-0: SidebarInset is a flex-row item (SidebarProvider), so without
          it wide message content (long unbroken strings, code blocks) propagates
          its min-content width and pushes the page past the viewport instead of
          wrapping/scrolling. */}
      <SidebarInset className="flex h-svh min-w-0 flex-col overflow-hidden">
        <header className="flex h-12 shrink-0 items-center px-2">
          <SidebarTrigger />
        </header>
        <main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
          {children}
        </main>
      </SidebarInset>
    </SidebarProvider>
  )
}
