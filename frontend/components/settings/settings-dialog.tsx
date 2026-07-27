"use client"

import * as React from "react"
import { useTheme } from "next-themes"

import { SETTINGS, type SettingDef } from "@/lib/settings"
import { useSettings } from "./settings-provider"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"

/**
 * The Ctrl/⌘+K command modal. Renders one control per entry in `SETTINGS`, so
 * new settings appear here automatically. Mount once (in the root layout).
 */
export function SettingsDialog() {
  const { commandOpen, setCommandOpen } = useSettings()

  // Global Ctrl/⌘+K to toggle. A ref tracks the latest open state so the
  // listener can flip it without re-subscribing on every change.
  const openRef = React.useRef(commandOpen)
  React.useEffect(() => {
    openRef.current = commandOpen
  }, [commandOpen])
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
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
      <DialogContent className="max-w-lg gap-5">
        <DialogHeader>
          {/* The ⌘K badge used to sit at the end of a `justify-between` row, in
              the same corner the dialog absolutely positions its close button —
              so the two overlapped. It reads as a hint about the title anyway,
              so it belongs beside it. */}
          <DialogTitle className="flex items-center gap-2">
            Settings
            <kbd className="rounded border border-border bg-muted px-1.5 py-0.5 font-mono text-xs font-normal text-muted-foreground">
              ⌘K
            </kbd>
          </DialogTitle>
          <DialogDescription>
            Tune how the assistant behaves. Toggles are also available as slash
            commands in the composer.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col divide-y divide-border">
          <ThemeRow />
          {SETTINGS.map((setting) => (
            <SettingRow key={setting.key} setting={setting} />
          ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}

/**
 * Theme, which is deliberately *not* a `SETTINGS` entry.
 *
 * next-themes already owns this value and persists it under its own key, and it
 * has to, because the no-flash script reads it before React exists. Declaring it
 * a second time in `Settings` would give the app two answers to what the theme
 * is. So it gets a hand-written row that reads and writes the real one.
 */
function ThemeRow() {
  // `theme` is undefined until next-themes has read storage, so no option marks
  // itself active on the server or the first client render — which is what we
  // want, and why this needs no mounted flag: the two renders already agree.
  const { theme, setTheme } = useTheme()

  const options = [
    ["light", "Light"],
    ["dark", "Dark"],
    ["system", "System"],
  ] as const

  return (
    <div className="flex items-start justify-between gap-4 py-3.5">
      <div className="flex flex-col gap-0.5">
        <span className="text-sm font-medium text-foreground">Theme</span>
        <span className="text-xs leading-relaxed text-muted-foreground">
          Warm paper, or Kanagawa Dragon after dark. Press D anywhere to flip.
        </span>
      </div>
      <div className="inline-flex shrink-0 gap-0.5 rounded-md border border-border p-0.5">
        {options.map(([value, label]) => (
          <Button
            key={value}
            type="button"
            size="xs"
            variant={theme === value ? "secondary" : "ghost"}
            onClick={() => setTheme(value)}
          >
            {label}
          </Button>
        ))}
      </div>
    </div>
  )
}

function SettingRow({ setting }: { setting: SettingDef }) {
  const { settings, setSetting } = useSettings()

  if (setting.kind === "toggle") {
    return (
      <div className="flex items-start justify-between gap-4 py-3.5">
        <div className="flex flex-col gap-0.5">
          <span className="text-sm font-medium text-foreground">
            {setting.label}
          </span>
          <span className="text-xs leading-relaxed text-muted-foreground">
            {setting.description}
          </span>
        </div>
        <Switch
          checked={settings[setting.key] as boolean}
          onCheckedChange={(checked) => setSetting(setting.key, checked)}
        />
      </div>
    )
  }

  const value = settings[setting.key] as string
  return (
    <div className="flex flex-col gap-2 py-3.5">
      <div className="flex flex-col gap-0.5">
        <span className="text-sm font-medium text-foreground">
          {setting.label}
        </span>
        <span className="text-xs leading-relaxed text-muted-foreground">
          {setting.description}
        </span>
      </div>
      {setting.multiline ? (
        <Textarea
          value={value}
          maxLength={setting.maxLength}
          placeholder={setting.placeholder}
          onChange={(e) => setSetting(setting.key, e.target.value)}
          className="min-h-24 resize-none text-sm"
        />
      ) : (
        <Input
          value={value}
          maxLength={setting.maxLength}
          placeholder={setting.placeholder}
          onChange={(e) => setSetting(setting.key, e.target.value)}
        />
      )}
    </div>
  )
}
