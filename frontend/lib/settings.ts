// The single source of truth for user settings.
//
// Every setting is declared once in `SETTINGS` below. The command modal
// (components/settings/settings-dialog.tsx) renders its controls from this list,
// and the slash-command runner (lib/commands.ts) derives its commands from the
// `toggle` entries. To add a setting you add ONE entry here (and a field on the
// `Settings` type) — it then shows up in the modal, under the section its `group`
// names, and, if it's a toggle, as a `/command` automatically. Nothing else
// needs touching.

export interface Settings {
  useLiveThrobber: boolean
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
  /**
   * Stretch the context window to this multiple of the model's trained
   * ctx_window, via NTK-aware RoPE scaling computed fresh for each turn. 1
   * (default) runs the model exactly as trained; above 1 buys headroom before
   * history starts getting evicted, at some quality cost on the tokens past
   * the trained length.
   */
  contextMultiplier: number
}

export type SettingGroup = "appearance" | "chat" | "decoding"

export const GROUPS: ReadonlyArray<{
  id: SettingGroup
  label: string
}> = [
  { id: "appearance", label: "Appearance" },
  { id: "chat", label: "Chat" },
  { id: "decoding", label: "Decoding" },
]

interface BaseSetting {
  key: keyof Settings
  group: SettingGroup
  label: string
  command?: string
}

/** An on/off setting. Rendered as a Switch, and exposed as a slash command. */
export interface ToggleSetting extends BaseSetting {
  kind: "toggle"
  default: boolean
  /** The slash-command word, e.g. "show-thinking" → `/show-thinking on|off`. */
  command: string
}

/** A free-text setting. Rendered as an input/textarea and `/system-prompt <text>`. */
export interface TextSetting extends BaseSetting {
  kind: "text"
  default: string
  placeholder?: string
  multiline?: boolean
  maxLength?: number
  command?: string
}

/**
 * A bounded numeric setting. Rendered as a slider with its value beside the
 * label and exposed as `/command <number>`.
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
  command?: string
}

export type SettingDef = ToggleSetting | TextSetting | NumberSetting

// Keep this in the order you want controls to appear in the modal.
export const SETTINGS: SettingDef[] = [
  {
    kind: "toggle",
    key: "useLiveThrobber",
    group: "appearance",
    label: "Use live throbber",
    default: false,
    command: "live-throbber",
  },
  {
    kind: "toggle",
    key: "showReasoning",
    group: "chat",
    label: "Show reasoning",
    default: false,
    command: "show-thinking",
  },
  {
    kind: "toggle",
    key: "notifyOnFinish",
    group: "chat",
    label: "Notify when a job finishes",
    default: false,
    command: "notify",
  },
  {
    kind: "text",
    key: "systemPrompt",
    group: "chat",
    label: "System prompt",
    default: "",
    placeholder: "You are a helpful assistant. Answer concisely.",
    multiline: true,
    maxLength: 2000,
    command: "system-prompt",
  },
  {
    kind: "number",
    key: "contextMultiplier",
    group: "chat",
    label: "Context multiplier",
    default: 1,
    min: 1,
    max: 8,
    step: 0.5,
    precision: 1,
    command: "context-multiplier",
  },
  {
    kind: "number",
    key: "temperature",
    group: "decoding",
    label: "Temperature",
    default: 0.8,
    min: 0.01,
    max: 2,
    step: 0.01,
    precision: 2,
    command: "temperature",
  },
  {
    kind: "number",
    key: "topK",
    group: "decoding",
    label: "Top-k",
    default: 40,
    min: 1,
    max: 200,
    step: 1,
    precision: 0,
    command: "top-k",
  },
  {
    kind: "number",
    key: "topP",
    group: "decoding",
    label: "Top-p",
    default: 0.9,
    min: 0.05,
    max: 1,
    step: 0.01,
    precision: 2,
    command: "top-p",
  },
]

export const DEFAULT_SETTINGS: Settings = Object.fromEntries(
  SETTINGS.map((s) => [s.key, s.default])
) as unknown as Settings

export const TOGGLE_SETTINGS = SETTINGS.filter(
  (s): s is ToggleSetting => s.kind === "toggle"
)
