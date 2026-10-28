"use client"

import * as React from "react"

import { GROUPS } from "@/lib/settings"
import { useAuth } from "@/hooks/use-auth"
import { useSettings } from "./settings-provider"
import { SettingsSection } from "./settings-section"
import { SettingsProfile } from "./settings-profile"
import { UsageOverview } from "@/components/usage/usage-overview"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"

const PANEL_CLASS =
  "min-w-0 overflow-y-auto px-4 py-8 data-ending-style:hidden sm:px-8"
const TAB_CLASS =
  "h-auto flex-none px-3 py-2 text-xs font-normal after:bg-primary group-data-vertical/tabs:after:inset-y-2 group-data-vertical/tabs:after:right-auto group-data-vertical/tabs:after:left-0 group-data-vertical/tabs:after:w-px"

export function SettingsDialog() {
  const { commandOpen, setCommandOpen, resetSettings, section, setSection } =
    useSettings()
  const { isAuthenticated } = useAuth()
  const activeSection =
    !isAuthenticated && (section === "profile" || section === "usage")
      ? "appearance"
      : section
  const close = () => setCommandOpen(false)

  const openRef = React.useRef(commandOpen)
  React.useEffect(() => {
    openRef.current = commandOpen
  }, [commandOpen])
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Password managers can dispatch keydowns without a string key.
      if (typeof e.key !== "string") return
      if (e.key.toLowerCase() === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        setCommandOpen(!openRef.current)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [setCommandOpen])

  return (
    <Dialog open={commandOpen} onOpenChange={setCommandOpen}>
      <DialogContent className="h-[min(36rem,calc(100dvh-2rem))] max-w-3xl gap-0 overflow-hidden bg-background p-0">
        <DialogTitle className="sr-only">Settings</DialogTitle>
        <DialogDescription className="sr-only">
          Manage your account, usage, and preferences.
        </DialogDescription>
        <Tabs
          orientation="vertical"
          value={activeSection}
          onValueChange={(value) => {
            if (
              value === "profile" ||
              value === "usage" ||
              GROUPS.some((group) => group.id === value)
            ) {
              setSection(value as typeof section)
            }
          }}
          className="min-h-0 flex-1 gap-0"
        >
          <aside className="flex w-28 shrink-0 flex-col border-r border-border px-2 py-8 sm:w-40 sm:px-4">
            <p className="mb-6 px-2 eyebrow text-muted-foreground">Settings</p>
            <TabsList
              aria-label="Settings sections"
              activateOnFocus
              variant="line"
              className="w-full items-stretch justify-start gap-1 p-0"
            >
              {GROUPS.map((group) => (
                <TabsTrigger
                  key={group.id}
                  value={group.id}
                  className={TAB_CLASS}
                >
                  {group.label}
                </TabsTrigger>
              ))}
              {isAuthenticated && (
                <>
                  <TabsTrigger value="usage" className={TAB_CLASS}>
                    Usage
                  </TabsTrigger>
                  <TabsTrigger value="profile" className={TAB_CLASS}>
                    Profile
                  </TabsTrigger>
                </>
              )}
            </TabsList>
            <kbd className="mt-auto px-2 pt-6 font-mono text-xs text-muted-foreground/60">
              ⌘K
            </kbd>
          </aside>
          {GROUPS.map((group) => (
            <TabsContent
              key={group.id}
              value={group.id}
              className={PANEL_CLASS}
            >
              <SettingsSection group={group} />
              <div className="mt-8 border-t border-border/60 pt-4">
                <Button
                  variant="ghost"
                  size="xs"
                  onClick={resetSettings}
                  className="text-muted-foreground"
                >
                  Reset to defaults
                </Button>
              </div>
            </TabsContent>
          ))}
          {isAuthenticated && (
            <>
              <TabsContent value="usage" className={PANEL_CLASS}>
                <h2 className="mb-6 font-display text-2xl">Usage</h2>
                <UsageOverview compact />
              </TabsContent>
              <TabsContent value="profile" className={PANEL_CLASS}>
                <SettingsProfile onClose={close} />
              </TabsContent>
            </>
          )}
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}
