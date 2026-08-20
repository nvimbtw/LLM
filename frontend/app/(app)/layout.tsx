import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"

/**
 * The signed-in frame: sidebar and scroll container. Everything except `/login`
 * and `/signup` renders inside it.
 *
 * There is no header row. It held one thing — a `SidebarTrigger` — and spent a
 * 48px band across every page in the app to do it, which on a page that opens
 * with a masthead reads as a gap above the title rather than as a bar. The
 * sidebar is `collapsible="icon"`, so it is never gone: collapsed it is still a
 * 3rem rail of glyphs, and `SidebarRail` (the strip along its edge) and ⌘/Ctrl+B
 * both toggle it. The button was the third way to do what two other affordances
 * already do, and the only one that charged the whole app for the privilege.
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
        <main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
          {children}
        </main>
      </SidebarInset>
    </SidebarProvider>
  )
}
