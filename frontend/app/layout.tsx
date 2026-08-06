import { Geist, Geist_Mono, Fraunces } from "next/font/google"

import "./globals.css"
import { ThemeProvider } from "@/components/theme-provider"
import { cn } from "@/lib/utils"

import {
  SidebarProvider,
  SidebarTrigger,
  SidebarInset,
} from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"
import { ChatsProvider } from "@/components/chat/chats-provider"
import { JobsProvider } from "@/components/jobs/jobs-provider"
import { ToastProvider } from "@/components/ui/toast"
import { SettingsProvider } from "@/components/settings/settings-provider"
import { SettingsDialog } from "@/components/settings/settings-dialog"

const geist = Geist({ subsets: ["latin"], variable: "--font-sans" })

const fontMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
})

// The display face. Fraunces is a variable serif with an optical-size axis, which
// is the whole reason it suits this: headings set large get the high-contrast,
// tight-fitting cut, and the same family at small sizes stays sturdy. `opsz` is
// left on `auto` so the browser tracks the rendered size.
//
// `SOFT` softens the terminals a little off the default, which takes the edge off
// the display cut without tipping into the decorative end of the family — the
// "wonk" axis is deliberately left alone.
const fontDisplay = Fraunces({
  subsets: ["latin"],
  variable: "--font-display",
  axes: ["SOFT"],
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
        "font-sans antialiased",
        geist.variable,
        fontMono.variable,
        fontDisplay.variable
      )}
    >
      <body>
        <ThemeProvider>
          <ToastProvider>
            <SettingsProvider>
              <SettingsDialog />
              <ChatsProvider>
                <JobsProvider>
                  <SidebarProvider>
                    <AppSidebar />
                    {/* h-svh + overflow-hidden bound the column so page content
                  scrolls inside <main>, keeping the header (and the chat
                  composer) pinned instead of scrolling the document body. */}
                    {/* min-w-0: SidebarInset is a flex-row item (SidebarProvider),
                  so without it wide message content (long unbroken strings,
                  code blocks) propagates its min-content width and pushes the
                  page past the viewport instead of wrapping/scrolling. */}
                    <SidebarInset className="flex h-svh min-w-0 flex-col overflow-hidden">
                      <header className="flex h-12 shrink-0 items-center px-2">
                        <SidebarTrigger />
                      </header>
                      <main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
                        {children}
                      </main>
                    </SidebarInset>
                  </SidebarProvider>
                </JobsProvider>
              </ChatsProvider>
            </SettingsProvider>
          </ToastProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}
