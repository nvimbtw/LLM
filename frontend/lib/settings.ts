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
  /**
   * Toast when a queued job finishes. Off by default: on the chat-only path a
   * job is your own reply arriving, and announcing that is noise. It earns its
   * keep once you are running training and cycles, which is where the wait is
   * long enough to have gone and done something else.
   */
  notifyOnFinish: boolean
  /** Prepended to every conversation to steer the assistant (empty = none). */
  systemPrompt: string
  /**
   * Decoding randomness. Low is near-greedy and repetitive; high wanders. These
   * three compose in the order they are declared — temperature reshapes the
   * distribution, then top-k caps how many candidates survive, then top-p keeps
   * the shortest prefix of those that already holds `topP` of the probability.
   */
  temperature: number
  /** Hard cap on surviving candidates. */
  topK: number
  /** Nucleus mass to keep. 1 disables it and leaves the cut to top-k alone. */
  topP: number
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

/**
 * A bounded numeric setting. Rendered as a slider with its value beside the
 * label; no slash command, since "/temperature 0.85" is worse than a drag.
 *
 * `min`/`max` mirror the range `routes/chat.rs::resolve_sampling` enforces — the
 * backend rejects rather than clamps, so a control that could produce a value it
 * refuses would be a UI that generates its own errors.
 */
export interface NumberSetting extends BaseSetting {
  kind: "number"
  default: number
  min: number
  max: number
  step: number
  /** Decimals to show beside the label. Integers pass 0. */
  precision: number
}

export type SettingDef = ToggleSetting | TextSetting | NumberSetting

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
    kind: "toggle",
    key: "notifyOnFinish",
    label: "Notify when a job finishes",
    description:
      "Raise a toast as each queued job lands, including the ones a cycle queued for you.",
    default: false,
    command: "notify",
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
  {
    kind: "number",
    key: "temperature",
    label: "Temperature",
    description:
      "How much randomness to allow when picking each token. Near 0 is almost greedy and tends to repeat itself; above 1 wanders.",
    default: 0.8,
    min: 0.01,
    max: 2,
    step: 0.01,
    precision: 2,
  },
  {
    kind: "number",
    key: "topK",
    label: "Top-k",
    description:
      "Only ever consider this many of the highest-scoring tokens. A hard cap, applied before top-p.",
    default: 40,
    min: 1,
    max: 200,
    step: 1,
    precision: 0,
  },
  {
    kind: "number",
    key: "topP",
    label: "Top-p",
    description:
      "Keep only the likeliest tokens that together hold this much of the probability, discarding the tail. 1 turns it off.",
    default: 0.9,
    min: 0.05,
    max: 1,
    step: 0.01,
    precision: 2,
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
