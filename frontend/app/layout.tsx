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
              <SidebarInset className="flex flex-col">
                <header className="flex h-12 shrink-0 items-center px-2">
                  <SidebarTrigger />
                </header>
                {children}
              </SidebarInset>
            </SidebarProvider>
          </ChatsProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}
