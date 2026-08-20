"use client"

import * as React from "react"
import { useTheme } from "next-themes"

import { GROUPS, SETTINGS, type SettingDef } from "@/lib/settings"
import { useSettings } from "./settings-provider"
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Slider } from "@/components/ui/slider"

/**
 * The Ctrl/⌘+K command modal. Renders one control per entry in `SETTINGS`,
 * bucketed into the bands `GROUPS` declares, so a new setting appears here
 * automatically and in the right place. Mount once (in the root layout).
 *
 * Three bands rather than one list: seven rows given identical treatment made
 * appearance, chat and the decoding knobs look like seven unrelated preferences,
 * when the last three are one mechanism in three parts.
 *
 * The header and footer are fixed and only the bands scroll. Scrolling the whole
 * card took the title off the top of the screen, which is how you end up three
 * sliders deep in a modal with nothing on screen saying what you are in.
 */
export function SettingsDialog() {
  const { commandOpen, setCommandOpen, resetSettings } = useSettings()

  // Global Ctrl/⌘+K to toggle. A ref tracks the latest open state so the
  // listener can flip it without re-subscribing on every change.
  const openRef = React.useRef(commandOpen)
  React.useEffect(() => {
    openRef.current = commandOpen
  }, [commandOpen])
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Guarded because `key` is not always a string despite its type: autofill
      // and password managers dispatch synthetic keydowns without it. See the
      // same guard in `theme-provider.tsx`.
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
      {/* `p-0` and `gap-0`: this dialog lays out its own three bands, and the
          card's default padding would put the header's rule inside a margin
          instead of across the card. */}
      <DialogContent className="max-w-2xl gap-0 p-0">
        <header className="border-b border-border px-6 py-4">
          {/* The ⌘K badge used to sit at the end of a `justify-between` row, in
              the same corner the dialog absolutely positions its close button —
              so the two overlapped. It reads as a hint about the title anyway,
              so it belongs beside it. */}
          <DialogTitle className="flex items-center gap-2">
            Settings
            <kbd className="rounded-(--radius-marker) border border-border bg-muted px-1.5 py-0.5 font-mono text-xs font-normal text-muted-foreground">
              ⌘K
            </kbd>
          </DialogTitle>
          <DialogDescription className="mt-1 text-xs">
            Kept on this device. Toggles are also available as slash commands in
            the composer.
          </DialogDescription>
        </header>

        {/* 70vh rather than a fixed height: the card is centred by transform, so
            it can grow to its content on a tall screen and only scroll when it
            genuinely does not fit. */}
        <div className="max-h-[70vh] overflow-y-auto px-6">
          {GROUPS.map((group) => {
            const rows = SETTINGS.filter((s) => s.group === group.id)
            if (group.id !== "appearance" && rows.length === 0) return null

            return (
              <section
                key={group.id}
                className="border-b border-border py-5 last:border-0 last:pb-6"
              >
                <h3 className="eyebrow text-muted-foreground">
                  {group.label}
                </h3>
                {group.blurb && (
                  <p className="mt-1.5 max-w-prose text-xs leading-relaxed text-muted-foreground/80">
                    {group.blurb}
                  </p>
                )}

                <div className="mt-3 flex flex-col gap-4">
                  {group.id === "appearance" && <ThemeRow />}
                  {rows.map((setting) => (
                    <SettingRow key={setting.key} setting={setting} />
                  ))}
                </div>
              </section>
            )
          })}
        </div>

        <footer className="flex items-center justify-between gap-3 border-t border-border px-6 py-3.5">
          {/* `resetSettings` has existed on the provider since it was written and
              nothing called it — so a modal of sliders had no way back to the
              defaults except knowing what they were. */}
          <Button variant="ghost" size="sm" onClick={resetSettings}>
            Reset to defaults
          </Button>
          <DialogClose render={<Button size="sm">Done</Button>} />
        </footer>
      </DialogContent>
    </Dialog>
  )
}

/**
 * The shared geometry of every row: the words on the left at a readable measure,
 * the control right-aligned in a column of its own.
 *
 * One rhythm for all four kinds is the point. Previously toggles were a
 * two-column row while sliders and the textarea stacked full-width under their
 * prose, so the modal alternated short and tall and read as a pile rather than a
 * list. `control` on its own line is the one deliberate exception — a system
 * prompt needs the width.
 */
function Row({
  label,
  description,
  control,
  below,
  htmlFor,
}: {
  label: string
  description: string
  control?: React.ReactNode
  below?: React.ReactNode
  htmlFor?: string
}) {
  return (
    <div>
      <div className="flex items-start justify-between gap-6">
        <div className="min-w-0">
          <label
            htmlFor={htmlFor}
            className="text-sm font-medium text-foreground"
          >
            {label}
          </label>
          <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">
            {description}
          </p>
        </div>
        {control && <div className="shrink-0 pt-0.5">{control}</div>}
      </div>
      {below && <div className="mt-2">{below}</div>}
    </div>
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
    <Row
      label="Theme"
      description="Warm paper, or Kanagawa Dragon after dark. Press D anywhere to flip."
      control={
        <div className="inline-flex gap-0.5 rounded-(--radius-control-sm) border border-border p-0.5">
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

function SettingRow({ setting }: { setting: SettingDef }) {
  const { settings, setSetting } = useSettings()
  const id = `setting-${setting.key}`

  if (setting.kind === "toggle") {
    return (
      <Row
        htmlFor={id}
        label={setting.label}
        description={setting.description}
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
      <Row
        label={setting.label}
        description={setting.description}
        control={
          // The number sits in the control column, above the track rather than
          // under the thumb: a value that moves with the drag is unreadable
          // while dragging, which is the only time anyone is watching it.
          <div className="w-40">
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
    <Row
      htmlFor={id}
      label={setting.label}
      description={setting.description}
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
