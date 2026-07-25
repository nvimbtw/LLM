"use client"

import * as React from "react"

import { SETTINGS, type SettingDef } from "@/lib/settings"
import { useSettings } from "./settings-provider"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
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
          <div className="flex items-center justify-between gap-3">
            <DialogTitle>Settings</DialogTitle>
            <kbd className="rounded border border-border bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
              ⌘K
            </kbd>
          </div>
          <DialogDescription>
            Tune how the assistant behaves. Toggles are also available as slash
            commands in the composer.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col divide-y divide-border/60">
          {SETTINGS.map((setting) => (
            <SettingRow key={setting.key} setting={setting} />
          ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}

function SettingRow({ setting }: { setting: SettingDef }) {
  const { settings, setSetting } = useSettings()

  if (setting.kind === "toggle") {
    return (
      <div className="flex items-start justify-between gap-4 py-3.5">
        <div className="flex flex-col gap-0.5">
          <span className="text-sm font-medium text-foreground">{setting.label}</span>
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
        <span className="text-sm font-medium text-foreground">{setting.label}</span>
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
