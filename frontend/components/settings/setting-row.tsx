"use client"

import type { SettingDef } from "@/lib/settings"
import { Switch } from "@/components/ui/switch"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Slider } from "@/components/ui/slider"
import { useSettings } from "./settings-provider"
import { SettingsRowLayout } from "./settings-row-layout"

export interface SettingRowProps {
  setting: SettingDef
}

export function SettingRow({ setting }: SettingRowProps) {
  const { settings, setSetting } = useSettings()
  const id = `setting-${setting.key}`

  if (setting.kind === "toggle") {
    return (
      <SettingsRowLayout
        htmlFor={id}
        label={setting.label}
        control={
          <Switch
            id={id}
            checked={settings[setting.key] as boolean}
            onCheckedChange={(checked) => setSetting(setting.key, checked)}
          />
        }
      />
    )
  }

  if (setting.kind === "number") {
    const current = settings[setting.key] as number
    return (
      <SettingsRowLayout
        label={setting.label}
        control={
          // The number sits in the control column, above the track rather than
          // under the thumb: a value that moves with the drag is unreadable
          // while dragging, which is the only time anyone is watching it.
          <div className="w-40 max-w-full">
            <div className="text-right font-mono text-xs text-muted-foreground tabular-nums">
              {current.toFixed(setting.precision)}
            </div>
            <Slider
              value={current}
              min={setting.min}
              max={setting.max}
              step={setting.step}
              aria-label={setting.label}
              // The primitive supports multi-thumb and hands back an array for
              // it; this is always single-thumb, so unwrap rather than widen
              // the setting.
              onValueChange={(next) =>
                setSetting(setting.key, Array.isArray(next) ? next[0] : next)
              }
              className="mt-1.5"
            />
          </div>
        }
      />
    )
  }

  const value = settings[setting.key] as string
  return (
    <SettingsRowLayout
      htmlFor={id}
      label={setting.label}
      below={
        setting.multiline ? (
          <Textarea
            id={id}
            value={value}
            maxLength={setting.maxLength}
            placeholder={setting.placeholder}
            onChange={(e) => setSetting(setting.key, e.target.value)}
            className="min-h-20 resize-none text-sm"
          />
        ) : (
          <Input
            id={id}
            value={value}
            maxLength={setting.maxLength}
            placeholder={setting.placeholder}
            onChange={(e) => setSetting(setting.key, e.target.value)}
          />
        )
      }
    />
  )
}
