import { Geist, Geist_Mono, Unbounded } from "next/font/google"

import "./globals.css"
import { ThemeProvider } from "@/components/theme-provider"
import { cn } from "@/lib/utils";

import { SidebarProvider, SidebarTrigger, SidebarInset } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { ChatsProvider } from "@/components/chat/chats-provider";

const geist = Geist({ subsets: ['latin'], variable: '--font-sans' })

const fontMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
})

const fontDisplay = Unbounded({
  subsets: ["latin"],
  variable: "--font-display",
})

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={cn(
        "antialiased font-sans",
        geist.variable,
        fontMono.variable,
        fontDisplay.variable
      )}
    >
      <body>
        <ThemeProvider>
          <ChatsProvider>
            <SidebarProvider>
              <AppSidebar />
              {/* h-svh + overflow-hidden bound the column so page content
                  scrolls inside <main>, keeping the header (and the chat
                  composer) pinned instead of scrolling the document body. */}
              <SidebarInset className="flex h-svh flex-col overflow-hidden">
                <header className="flex h-12 shrink-0 items-center px-2">
                  <SidebarTrigger />
                </header>
                <main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
                  {children}
                </main>
              </SidebarInset>
            </SidebarProvider>
          </ChatsProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}
