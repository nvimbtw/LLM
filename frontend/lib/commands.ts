import { toast } from "@/components/ui/toast"
import { SETTINGS, TOGGLE_SETTINGS, type Settings } from "@/lib/settings"

// Slash commands mirror the settings that don't need free-text input. Every
// `toggle` setting becomes `/<command> [on|off|toggle]` automatically (no arg
// flips it), plus two built-ins: `/settings` opens the modal and `/help` lists
// what's available. Text settings (the system prompt) are intentionally modal-only.

export interface CommandContext {
  settings: Settings
  setSetting: <K extends keyof Settings>(key: K, value: Settings[K]) => void
  openSettings: () => void
}

const ON = new Set(["on", "true", "yes", "1", "enable", "enabled"])
const OFF = new Set(["off", "false", "no", "0", "disable", "disabled"])

/**
 * The command name if `text` is a *standalone, recognised* command — i.e. the
 * whole message is just `/<known> [args]` on a single line. A "/" mid-sentence,
 * a multi-line message, or an unknown word returns null, so those are treated as
 * ordinary prompts rather than commands.
 */
export function matchStandaloneCommand(text: string): string | null {
  const t = text.trim()
  if (!t.startsWith("/") || t.includes("\n")) return null
  const word = t.slice(1).split(/\s+/)[0].toLowerCase()
  return COMMAND_NAMES.includes(word) ? word : null
}

/**
 * Interpret a slash command. Returns true if the input was a command (handled or
 * rejected), meaning the caller must NOT forward it to the model. Every path
 * surfaces a toast so the user gets feedback.
 */
export function runSlashCommand(input: string, ctx: CommandContext): boolean {
  const trimmed = input.trim()
  if (!trimmed.startsWith("/")) return false

  const [rawName, ...rest] = trimmed.slice(1).split(/\s+/)
  const name = rawName.toLowerCase()
  const arg = rest.join(" ").toLowerCase().trim()

  if (name === "settings") {
    ctx.openSettings()
    return true
  }

  if (name === "help") {
    const list = TOGGLE_SETTINGS.map((s) => `/${s.command} on|off`).join(" · ")
    toast({ title: "Commands", description: `${list} · /settings` })
    return true
  }

  const setting = TOGGLE_SETTINGS.find((s) => s.command === name)
  if (setting) {
    const current = ctx.settings[setting.key] as boolean
    let next: boolean
    if (arg === "" || arg === "toggle") next = !current
    else if (ON.has(arg)) next = true
    else if (OFF.has(arg)) next = false
    else {
      toast.error(`/${name}`, `Use "on", "off", or nothing to toggle.`)
      return true
    }
    ctx.setSetting(setting.key, next)
    toast({
      title: setting.label,
      description: next ? "On" : "Off",
      type: "success",
    })
    return true
  }

  toast.error("Unknown command", `“/${name}” isn't a command. Try /help.`)
  return true
}

// Exposed for any future command palette / autocomplete over the composer.
export const COMMAND_NAMES = [
  "settings",
  "help",
  ...SETTINGS.filter((s) => s.kind === "toggle").map(
    (s) => (s as { command: string }).command
  ),
]
