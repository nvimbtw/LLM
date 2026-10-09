"use client"

import { useTheme } from "next-themes"
import { Button } from "@/components/ui/button"
import { SettingsRowLayout } from "./settings-row-layout"

/**
 * Theme, which is deliberately *not* a `SETTINGS` entry.
 *
 * next-themes already owns this value and persists it under its own key, and it
 * has to, because the no-flash script reads it before React exists. Declaring it
 * a second time in `Settings` would give the app two answers to what the theme
 * is. So it gets a hand-written row that reads and writes the real one.
 */
export function ThemeRow() {
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
    <SettingsRowLayout
      label="Theme"
      control={
        <div className="inline-flex max-w-full flex-wrap gap-0.5 rounded-(--radius-control-sm) border border-border p-0.5">
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
      }
    />
  )
}
