"use client"

import * as React from "react"

import { GROUPS } from "@/lib/settings"
import { useAuth } from "@/hooks/use-auth"
import { useSettings } from "./settings-provider"
import { SettingsSection } from "./settings-section"
import { SettingsProfile } from "./settings-profile"
import { UsageOverview } from "@/components/usage/usage-overview"
import {
  AccountModal,
  AccountModalRail,
  ACCOUNT_MODAL_PANEL as PANEL_CLASS,
  ACCOUNT_MODAL_TAB as TAB_CLASS,
} from "@/components/ui/account-modal"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"

/** Presents account settings and profile sections in the shared account modal. */
export function SettingsDialog() {
  const { commandOpen, setCommandOpen, resetSettings, section, setSection } =
    useSettings()
  const { isAuthenticated } = useAuth()
  const activeSection =
    !isAuthenticated && (section === "profile" || section === "usage")
      ? "appearance"
      : section
  /** Closes the modal after an account action completes. */
  const close = () => setCommandOpen(false)

  const openRef = React.useRef(commandOpen)
  React.useEffect(() => {
    openRef.current = commandOpen
  }, [commandOpen])
  React.useEffect(() => {
    /** Toggles settings using the application shortcut, ignoring malformed synthetic key events. */
    const onKey = (e: KeyboardEvent) => {
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
    <AccountModal
      open={commandOpen}
      onOpenChange={setCommandOpen}
      title="Settings"
      description="Manage your account, usage, and preferences."
    >
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
        <AccountModalRail title="Settings" shortcut="⌘K">
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
        </AccountModalRail>
        {GROUPS.map((group) => (
          <TabsContent key={group.id} value={group.id} className={PANEL_CLASS}>
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
    </AccountModal>
  )
}
