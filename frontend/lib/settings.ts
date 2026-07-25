// The single source of truth for user settings.
//
// Every setting is declared once in `SETTINGS` below. The command modal
// (components/settings/settings-dialog.tsx) renders its controls from this list,
// and the slash-command runner (lib/commands.ts) derives its commands from the
// `toggle` entries. To add a setting you add ONE entry here (and a field on the
// `Settings` type) — it then shows up in the modal, and, if it's a toggle, as a
// `/command` automatically. Nothing else needs touching.

export interface Settings {
  /**
   * Reveal the model's full reasoning as it thinks. Off: just the "Thinking…"
   * animation. On: the streamed reasoning text shown above the reply too.
   */
  showReasoning: boolean
  /** Prepended to every conversation to steer the assistant (empty = none). */
  systemPrompt: string
}

interface BaseSetting {
  key: keyof Settings
  label: string
  description: string
}

/** An on/off setting. Rendered as a Switch, and exposed as a slash command. */
export interface ToggleSetting extends BaseSetting {
  kind: "toggle"
  default: boolean
  /** The slash-command word, e.g. "thinking" → `/thinking on|off`. */
  command: string
}

/** A free-text setting. Rendered as an input/textarea; no slash command. */
export interface TextSetting extends BaseSetting {
  kind: "text"
  default: string
  placeholder?: string
  multiline?: boolean
  maxLength?: number
}

export type SettingDef = ToggleSetting | TextSetting

// Keep this in the order you want controls to appear in the modal.
export const SETTINGS: SettingDef[] = [
  {
    kind: "toggle",
    key: "showReasoning",
    label: "Show reasoning",
    description:
      "Reveal the model’s full reasoning above the reply as it thinks. The “Thinking…” animation shows either way.",
    default: false,
    command: "thinking",
  },
  {
    kind: "text",
    key: "systemPrompt",
    label: "System prompt",
    description: "Sent ahead of every conversation to steer the assistant.",
    default: "",
    placeholder: "You are a helpful assistant. Answer concisely.",
    multiline: true,
    maxLength: 2000,
  },
]

export const DEFAULT_SETTINGS: Settings = SETTINGS.reduce((acc, s) => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(acc as any)[s.key] = s.default
  return acc
}, {} as Settings)

export const TOGGLE_SETTINGS = SETTINGS.filter(
  (s): s is ToggleSetting => s.kind === "toggle"
)
