import type { Metadata } from "next"
import { Geist, Geist_Mono, Fraunces } from "next/font/google"

import "./globals.css"
import { ThemeProvider } from "@/components/theme-provider"
import { cn } from "@/lib/utils"

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

export const metadata: Metadata = {
  title: { default: "Slate", template: "%s · Slate" },
}

/**
 * Fonts, theme and the providers — and nothing else.
 *
 * The sidebar chrome used to live here, which meant `/login` and `/signup`
 * rendered inside it: a logged-out visitor got the app frame, the sidebar
 * toggle, and a nav for pages they could not open. It moved down into
 * `(app)/layout.tsx`, and `(auth)/layout.tsx` renders bare. Route groups do not
 * appear in the URL, so every path is what it was.
 *
 * The providers stay above the split on purpose. `ChatsProvider` already keys
 * its load off the session, so it handles being mounted logged-out, and keeping
 * it here means logging in does not remount the tree that just fetched.
 */
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
                <JobsProvider>{children}</JobsProvider>
              </ChatsProvider>
            </SettingsProvider>
          </ToastProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}
