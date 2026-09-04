import {
  Bell,
  Cpu,
  Eraser,
  Expand,
  Gauge,
  HelpCircle,
  ListFilter,
  MessageSquarePlus,
  Percent,
  Scissors,
  ScrollText,
  Settings2,
  SunMoon,
  Thermometer,
  Eye,
  type LucideIcon,
} from "lucide-react"

import { toast } from "@/components/ui/toast"
import {
  SETTINGS,
  type NumberSetting,
  type Settings,
  type TextSetting,
} from "@/lib/settings"
import type { Model } from "@/hooks/use-models"
import { THINK_BUDGETS, type ThinkBudget } from "@/lib/think-budget"

export interface CommandContext {
  settings: Settings
  setSetting: <K extends keyof Settings>(key: K, value: Settings[K]) => void
  openSettings: () => void
  model?: string
  setModel?: (id: string) => void
  models?: Model[]
  budget?: ThinkBudget
  setBudget?: (budget: ThinkBudget) => void
  theme?: string
  setTheme?: (theme: string) => void
  clearComposer?: () => void
  /** `/new` — navigates to the empty composer. Present everywhere `ChatComposer`
   *  mounts; on the new-chat screen itself it's a no-op navigation to where you
   *  already are, the same as clicking the sidebar's own "New" link twice. */
  newChat?: () => void
  /** `/prune` — see `lib/context-floor.ts`. Absent on the new-chat composer,
   *  which has no chat yet to prune. */
  pruneContext?: () => Promise<{ prunedTurns: number; prunedTokens: number }>
}

export type CommandKind =
  | "model"
  | "budget"
  | "toggle"
  | "number"
  | "text"
  | "theme"
  | "action"

export interface SlashCommandDef {
  name: string
  aliases: string[]
  label: string
  description: string
  argsHint: string
  kind: CommandKind
  settingKey?: keyof Settings
  icon: LucideIcon
}

export const SLASH_COMMANDS: SlashCommandDef[] = [
  {
    name: "model",
    aliases: ["m"],
    label: "Model",
    description: "Switch the active model for this chat",
    argsHint: "<model_name>",
    kind: "model",
    icon: Cpu,
  },
  {
    name: "thinking",
    aliases: ["budget", "think"],
    label: "Thinking Budget",
    description: "Set how long the model reasons before answering",
    argsHint: "[auto|low|med|high]",
    kind: "budget",
    icon: Gauge,
  },
  {
    name: "show-thinking",
    aliases: ["reasoning", "thinking-view", "show-reasoning"],
    label: "Show Reasoning",
    description: "Reveal the model’s streamed reasoning in the chat",
    argsHint: "[on|off]",
    kind: "toggle",
    settingKey: "showReasoning",
    icon: Eye,
  },
  {
    name: "temperature",
    aliases: ["temp", "t"],
    label: "Temperature",
    description:
      "Decoding randomness (0.01 – 2.00). Near 0 is greedy, > 1 wanders",
    argsHint: "<0.01 – 2.00>",
    kind: "number",
    settingKey: "temperature",
    icon: Thermometer,
  },
  {
    name: "top-k",
    aliases: ["topk", "k"],
    label: "Top-k",
    description: "Hard cap on surviving candidate tokens (1 – 200)",
    argsHint: "<1 – 200>",
    kind: "number",
    settingKey: "topK",
    icon: ListFilter,
  },
  {
    name: "top-p",
    aliases: ["topp", "p"],
    label: "Top-p",
    description: "Nucleus probability mass to keep (0.05 – 1.00)",
    argsHint: "<0.05 – 1.00>",
    kind: "number",
    settingKey: "topP",
    icon: Percent,
  },
  {
    name: "context-multiplier",
    aliases: ["multiplier", "ctx-multiplier", "context-window"],
    label: "Context Multiplier",
    description:
      "Stretch the context window past what the model trained on (1 – 8x). Quality past the trained length is a step down",
    argsHint: "<1 – 8>",
    kind: "number",
    settingKey: "contextMultiplier",
    icon: Expand,
  },
  {
    name: "prune",
    aliases: ["trim-context", "prune-context"],
    label: "Prune Context",
    description:
      "Cut this chat back to what fits its trained context window, keeping the multiplier for new headroom. No-op at 1×",
    argsHint: "",
    kind: "action",
    icon: Scissors,
  },
  {
    name: "system-prompt",
    aliases: ["system", "sysprompt", "prompt"],
    label: "System Prompt",
    description: "Prepend steering instructions to every conversation",
    argsHint: "<prompt>",
    kind: "text",
    settingKey: "systemPrompt",
    icon: ScrollText,
  },
  {
    name: "notify",
    aliases: ["notifications", "notify-on-finish"],
    label: "Notify on Finish",
    description: "Raise a toast when background queued jobs complete",
    argsHint: "[on|off]",
    kind: "toggle",
    settingKey: "notifyOnFinish",
    icon: Bell,
  },
  {
    name: "theme",
    aliases: ["mode", "dark", "light"],
    label: "Theme",
    description: "Switch appearance (light warm paper / dark Dragon / system)",
    argsHint: "[light|dark|system]",
    kind: "theme",
    icon: SunMoon,
  },
  {
    name: "settings",
    aliases: ["config", "preferences"],
    label: "Settings",
    description: "Open the full settings modal (⌘K)",
    argsHint: "",
    kind: "action",
    icon: Settings2,
  },
  {
    name: "help",
    aliases: ["commands", "?"],
    label: "Help",
    description: "Show available slash commands summary",
    argsHint: "",
    kind: "action",
    icon: HelpCircle,
  },
  {
    name: "clear",
    aliases: ["reset"],
    label: "Clear",
    description: "Clear the current composer input",
    argsHint: "",
    kind: "action",
    icon: Eraser,
  },
  {
    name: "new",
    aliases: ["n", "new-chat"],
    label: "New Chat",
    description: "Start a new conversation",
    argsHint: "",
    kind: "action",
    icon: MessageSquarePlus,
  },
]

export const COMMAND_NAMES = SLASH_COMMANDS.map((c) => c.name)

export function findCommand(nameOrAlias: string): SlashCommandDef | undefined {
  const norm = nameOrAlias.toLowerCase().trim().replace(/^\//, "")
  return SLASH_COMMANDS.find((c) => c.name === norm || c.aliases.includes(norm))
}

export function matchStandaloneCommand(text: string): SlashCommandDef | null {
  const t = text.trim()
  if (!t.startsWith("/") || t.includes("\n")) return null
  const word = t.slice(1).split(/\s+/)[0]
  return findCommand(word) ?? null
}

// ---------------------------------------------------------------------------
// Fuzzy matching algorithm
// ---------------------------------------------------------------------------

export interface FuzzyResult {
  match: boolean
  score: number
  ranges: [number, number][]
}

/**
 * High-precision fuzzy match scoring.
 * Favors exact matches, word-boundary/prefix hits, and consecutive characters.
 */
export function fuzzyMatch(pattern: string, target: string): FuzzyResult {
  const p = pattern.toLowerCase()
  const t = target.toLowerCase()

  if (!p) return { match: true, score: 0, ranges: [] }
  if (p === t) return { match: true, score: 1000, ranges: [[0, target.length]] }

  let pIdx = 0
  let tIdx = 0
  let score = 0
  let consecutive = 0
  const ranges: [number, number][] = []
  let rangeStart = -1

  while (pIdx < p.length && tIdx < t.length) {
    if (p[pIdx] === t[tIdx]) {
      if (rangeStart === -1) rangeStart = tIdx

      let charScore = 10
      // Prefix bonus
      if (tIdx === 0) charScore += 40
      // Word boundary bonus (after space, slash, dash, underscore)
      else if (/[\s/_\-.]/.test(t[tIdx - 1])) charScore += 30

      // Consecutive match multiplier
      charScore += consecutive * 15
      consecutive++
      score += charScore

      pIdx++
    } else {
      if (rangeStart !== -1) {
        ranges.push([rangeStart, tIdx])
        rangeStart = -1
      }
      consecutive = 0
    }
    tIdx++
  }

  if (pIdx === p.length) {
    if (rangeStart !== -1) ranges.push([rangeStart, tIdx])
    // Shorter targets with same match rank higher
    score -= (target.length - pattern.length) * 2
    return { match: true, score, ranges }
  }

  return { match: false, score: -1, ranges: [] }
}

// ---------------------------------------------------------------------------
// Suggestions Engine
// ---------------------------------------------------------------------------

export interface CommandSuggestion {
  command: SlashCommandDef
  score: number
  currentDisplay?: string
}

export interface ModelOption {
  model: Model
  isCurrent: boolean
}

export interface BudgetOption {
  value: ThinkBudget
  label: string
  hint: string
  isCurrent: boolean
}

export interface NumberRangeInfo {
  min: number
  max: number
  step: number
  precision: number
  current: number
  default: number
  presets: { label: string; value: number }[]
}

export interface TextPromptInfo {
  current: string
  placeholder?: string
  templates: { label: string; value: string }[]
}

export type ArgumentSuggestions =
  | { type: "model"; options: ModelOption[] }
  | { type: "budget"; options: BudgetOption[] }
  | {
      type: "toggle"
      options: { value: string; label: string; isCurrent: boolean }[]
    }
  | {
      type: "theme"
      options: {
        value: string
        label: string
        description: string
        isCurrent: boolean
      }[]
    }
  | { type: "number"; info: NumberRangeInfo }
  | { type: "text"; info: TextPromptInfo }
  | { type: "action"; description: string }

export type SuggestionResult =
  | { mode: "command"; query: string; suggestions: CommandSuggestion[] }
  | {
      mode: "argument"
      command: SlashCommandDef
      argQuery: string
      data: ArgumentSuggestions
    }

const NUMBER_PRESETS: Record<string, { label: string; value: number }[]> = {
  temperature: [
    { label: "0.20 (Focused)", value: 0.2 },
    { label: "0.70 (Balanced)", value: 0.7 },
    { label: "1.00 (Creative)", value: 1.0 },
    { label: "1.50 (Exploratory)", value: 1.5 },
  ],
  topK: [
    { label: "10 (Strict)", value: 10 },
    { label: "40 (Default)", value: 40 },
    { label: "100 (Broad)", value: 100 },
  ],
  topP: [
    { label: "0.50 (Focused)", value: 0.5 },
    { label: "0.90 (Default)", value: 0.9 },
    { label: "1.00 (Disabled)", value: 1.0 },
  ],
  contextMultiplier: [
    { label: "1x (Default)", value: 1 },
    { label: "2x", value: 2 },
    { label: "4x", value: 4 },
  ],
}

const SYSTEM_TEMPLATES = [
  {
    label: "Concise Assistant",
    value: "You are a helpful, concise assistant.",
  },
  {
    label: "Code Specialist",
    value:
      "You are an expert software engineer. Provide clean, well-tested code.",
  },
  {
    label: "Deep Reasoning",
    value: "Think through problems step-by-step before answering.",
  },
]

export function getCommandSuggestions(
  input: string,
  ctx: CommandContext
): SuggestionResult | null {
  const trimmed = input.trimStart()
  if (!trimmed.startsWith("/")) return null

  const content = trimmed.slice(1)
  const spaceIdx = content.indexOf(" ")

  // Stage 1: Searching for commands
  if (spaceIdx === -1) {
    const query = content.toLowerCase()
    // A bare "/" hasn't narrowed anything yet — dumping the full command list
    // is noise, not help. Wait for at least one character to fuzzy-match on.
    if (!query) return null

    const suggestions: CommandSuggestion[] = []

    for (const cmd of SLASH_COMMANDS) {
      let maxScore = -1

      // Match command name
      const nameMatch = fuzzyMatch(query, cmd.name)
      if (nameMatch.match && nameMatch.score > maxScore) {
        maxScore = nameMatch.score + 20
      }

      // Match aliases
      for (const alias of cmd.aliases) {
        const aliasMatch = fuzzyMatch(query, alias)
        if (aliasMatch.match && aliasMatch.score > maxScore) {
          maxScore = aliasMatch.score + 10
        }
      }

      // Match label & description
      const labelMatch = fuzzyMatch(query, cmd.label)
      if (labelMatch.match && labelMatch.score > maxScore) {
        maxScore = labelMatch.score
      }

      if (maxScore >= 0) {
        let currentDisplay: string | undefined
        if (cmd.kind === "model" && ctx.model) {
          const currentModel = ctx.models?.find((m) => m.id === ctx.model)
          currentDisplay = currentModel?.name ?? ctx.model
        } else if (cmd.kind === "budget" && ctx.budget) {
          currentDisplay = ctx.budget
        } else if (cmd.kind === "theme" && ctx.theme) {
          currentDisplay = ctx.theme
        } else if (cmd.settingKey && ctx.settings) {
          const val = ctx.settings[cmd.settingKey]
          if (typeof val === "boolean") currentDisplay = val ? "On" : "Off"
          else if (typeof val === "number") currentDisplay = val.toString()
          else if (typeof val === "string" && val)
            currentDisplay = `"${val.slice(0, 16)}…"`
        }

        suggestions.push({
          command: cmd,
          score: maxScore >= 0 ? maxScore : 0,
          currentDisplay,
        })
      }
    }

    suggestions.sort((a, b) => b.score - a.score)
    return { mode: "command", query, suggestions }
  }

  // Stage 2: In command argument mode
  const commandWord = content.slice(0, spaceIdx).toLowerCase()
  const argQuery = content.slice(spaceIdx + 1).trim()
  const command = findCommand(commandWord)

  if (!command) return null

  switch (command.kind) {
    case "model": {
      const models = ctx.models ?? []
      let filtered = models
      if (argQuery) {
        filtered = models
          .map((m) => {
            const nameScore = fuzzyMatch(argQuery, m.name).score
            const idScore = fuzzyMatch(argQuery, m.id).score
            const famScore = fuzzyMatch(argQuery, m.family || "").score
            return { model: m, score: Math.max(nameScore, idScore, famScore) }
          })
          .filter((item) => item.score >= 0)
          .sort((a, b) => b.score - a.score)
          .map((item) => item.model)
      }
      return {
        mode: "argument",
        command,
        argQuery,
        data: {
          type: "model",
          options: filtered.map((m) => ({
            model: m,
            isCurrent: m.id === ctx.model,
          })),
        },
      }
    }

    case "budget": {
      const currentBudget = ctx.budget ?? "auto"
      const options = THINK_BUDGETS.filter(
        (b) =>
          !argQuery ||
          b.value.includes(argQuery.toLowerCase()) ||
          b.label.toLowerCase().includes(argQuery.toLowerCase())
      ).map((b) => ({
        ...b,
        isCurrent: b.value === currentBudget,
      }))
      return {
        mode: "argument",
        command,
        argQuery,
        data: { type: "budget", options },
      }
    }

    case "toggle": {
      const isCurrent = command.settingKey
        ? (ctx.settings[command.settingKey] as boolean)
        : false
      const opts = [
        { value: "on", label: "Turn On", isCurrent },
        { value: "off", label: "Turn Off", isCurrent: !isCurrent },
        { value: "toggle", label: "Toggle State", isCurrent: false },
      ].filter((o) => !argQuery || o.value.includes(argQuery.toLowerCase()))
      return {
        mode: "argument",
        command,
        argQuery,
        data: { type: "toggle", options: opts },
      }
    }

    case "theme": {
      const currentTheme = ctx.theme || "system"
      const opts = [
        {
          value: "light",
          label: "Light",
          description: "Warm paper",
          isCurrent: currentTheme === "light",
        },
        {
          value: "dark",
          label: "Dark",
          description: "Kanagawa Dragon",
          isCurrent: currentTheme === "dark",
        },
        {
          value: "system",
          label: "System",
          description: "Follow system preference",
          isCurrent: currentTheme === "system",
        },
      ].filter(
        (o) =>
          !argQuery ||
          o.value.includes(argQuery.toLowerCase()) ||
          o.label.toLowerCase().includes(argQuery.toLowerCase())
      )
      return {
        mode: "argument",
        command,
        argQuery,
        data: { type: "theme", options: opts },
      }
    }

    case "number": {
      const settingDef = SETTINGS.find((s) => s.key === command.settingKey) as
        | NumberSetting
        | undefined
      if (!settingDef) return null
      const current = ctx.settings[command.settingKey!] as number
      return {
        mode: "argument",
        command,
        argQuery,
        data: {
          type: "number",
          info: {
            min: settingDef.min,
            max: settingDef.max,
            step: settingDef.step,
            precision: settingDef.precision,
            current,
            default: settingDef.default,
            presets: NUMBER_PRESETS[command.settingKey as string] || [],
          },
        },
      }
    }

    case "text": {
      const settingDef = SETTINGS.find((s) => s.key === command.settingKey) as
        | TextSetting
        | undefined
      const current = (ctx.settings[command.settingKey!] as string) || ""
      return {
        mode: "argument",
        command,
        argQuery,
        data: {
          type: "text",
          info: {
            current,
            placeholder: settingDef?.placeholder,
            templates: SYSTEM_TEMPLATES,
          },
        },
      }
    }

    case "action":
    default:
      return {
        mode: "argument",
        command,
        argQuery,
        data: { type: "action", description: command.description },
      }
  }
}

// ---------------------------------------------------------------------------
// Execution Logic
// ---------------------------------------------------------------------------

const ON = new Set(["on", "true", "yes", "1", "enable", "enabled"])
const OFF = new Set(["off", "false", "no", "0", "disable", "disabled"])

export function runSlashCommand(input: string, ctx: CommandContext): boolean {
  const trimmed = input.trim()
  if (!trimmed.startsWith("/")) return false

  const [rawName, ...rest] = trimmed.slice(1).split(/\s+/)
  const name = rawName.toLowerCase()
  const arg = rest.join(" ").trim()
  const command = findCommand(name)

  if (!command) {
    toast.error(
      "Unknown command",
      `“/${name}” isn't a command. Type / for suggestions.`
    )
    return true
  }

  switch (command.kind) {
    case "model": {
      if (!arg) {
        const current =
          ctx.models?.find((m) => m.id === ctx.model)?.name ?? ctx.model
        toast({
          title: "Current Model",
          description: current ? `Selected: ${current}` : "No model selected.",
        })
        return true
      }

      const models = ctx.models ?? []
      const exact = models.find(
        (m) =>
          m.id.toLowerCase() === arg.toLowerCase() ||
          m.name.toLowerCase() === arg.toLowerCase()
      )
      const target =
        exact ||
        models.find(
          (m) =>
            m.id.toLowerCase().includes(arg.toLowerCase()) ||
            m.name.toLowerCase().includes(arg.toLowerCase())
        )

      if (target) {
        ctx.setModel?.(target.id)
        toast({
          title: "Model switched",
          description: target.name,
          type: "success",
        })
      } else {
        toast.error("Model not found", `No model matching “${arg}”.`)
      }
      return true
    }

    case "budget": {
      const val = arg.toLowerCase()
      let budget: ThinkBudget | null = null
      if (val === "auto" || val === "none" || val === "default") budget = "auto"
      else if (val === "low" || val === "min" || val === "short") budget = "low"
      else if (val === "med" || val === "medium" || val === "mid")
        budget = "med"
      else if (val === "high" || val === "max" || val === "long")
        budget = "high"
      else if (!val) {
        // Cycle budget
        const curr = ctx.budget || "auto"
        const nextMap: Record<ThinkBudget, ThinkBudget> = {
          auto: "low",
          low: "med",
          med: "high",
          high: "auto",
        }
        budget = nextMap[curr]
      }

      if (budget) {
        ctx.setBudget?.(budget)
        const item = THINK_BUDGETS.find((b) => b.value === budget)
        toast({
          title: "Thinking Budget",
          description: item ? `${item.label} — ${item.hint}` : budget,
          type: "success",
        })
      } else {
        toast.error("Invalid budget", `Use "auto", "low", "med", or "high".`)
      }
      return true
    }

    case "toggle": {
      if (!command.settingKey) return true
      const current = ctx.settings[command.settingKey] as boolean
      let next: boolean
      const argLower = arg.toLowerCase()
      if (argLower === "" || argLower === "toggle") next = !current
      else if (ON.has(argLower)) next = true
      else if (OFF.has(argLower)) next = false
      else {
        toast.error(
          `/${command.name}`,
          `Use "on", "off", or nothing to toggle.`
        )
        return true
      }
      ctx.setSetting(command.settingKey, next)
      toast({
        title: command.label,
        description: next ? "Enabled" : "Disabled",
        type: "success",
      })
      return true
    }

    case "number": {
      if (!command.settingKey) return true
      const settingDef = SETTINGS.find((s) => s.key === command.settingKey) as
        | NumberSetting
        | undefined
      if (!settingDef) return true

      if (!arg) {
        const current = ctx.settings[command.settingKey] as number
        toast({
          title: command.label,
          description: `Current: ${current.toFixed(settingDef.precision)} (Range: ${settingDef.min} – ${settingDef.max})`,
        })
        return true
      }

      const num = Number.parseFloat(arg)
      if (Number.isNaN(num) || num < settingDef.min || num > settingDef.max) {
        toast.error(
          `Invalid ${command.label.toLowerCase()}`,
          `Value must be a number between ${settingDef.min} and ${settingDef.max}.`
        )
        return true
      }

      ctx.setSetting(command.settingKey, num)
      toast({
        title: command.label,
        description: num.toFixed(settingDef.precision),
        type: "success",
      })
      return true
    }

    case "text": {
      if (!command.settingKey) return true
      const lower = arg.toLowerCase()
      if (
        lower === "clear" ||
        lower === "reset" ||
        lower === "none" ||
        lower === '""'
      ) {
        ctx.setSetting(command.settingKey, "")
        toast({
          title: "System Prompt",
          description: "Cleared (no steering prompt)",
          type: "success",
        })
        return true
      }

      if (!arg) {
        const current = ctx.settings[command.settingKey] as string
        toast({
          title: "System Prompt",
          description: current
            ? `Current: “${current}”`
            : "Currently unset. Provide a prompt to set it.",
        })
        return true
      }

      ctx.setSetting(command.settingKey, arg)
      const preview = arg.length > 60 ? `${arg.slice(0, 60)}…` : arg
      toast({
        title: "System Prompt updated",
        description: `“${preview}”`,
        type: "success",
      })
      return true
    }

    case "theme": {
      const val = arg.toLowerCase()
      if (val === "light" || val === "dark" || val === "system") {
        ctx.setTheme?.(val)
        toast({
          title: "Theme updated",
          description: val.charAt(0).toUpperCase() + val.slice(1),
          type: "success",
        })
        return true
      }
      if (val === "toggle" || !val) {
        const next = ctx.theme === "dark" ? "light" : "dark"
        ctx.setTheme?.(next)
        toast({
          title: "Theme flipped",
          description: next === "dark" ? "Dark (Dragon)" : "Light (Warm Paper)",
          type: "success",
        })
        return true
      }
      toast.error("Theme", `Use "light", "dark", "system", or "toggle".`)
      return true
    }

    case "action": {
      if (command.name === "settings") {
        ctx.openSettings()
        return true
      }
      if (command.name === "clear") {
        ctx.clearComposer?.()
        return true
      }
      if (command.name === "new") {
        ctx.newChat?.()
        return true
      }
      if (command.name === "prune") {
        if (ctx.settings.contextMultiplier <= 1) {
          toast({
            title: "Prune Context",
            description: "Nothing to prune — the context multiplier is 1×.",
          })
          return true
        }
        // Async, unlike every other action here: it asks the engine where the
        // cut would land. The command is still handled synchronously — the
        // toast just arrives a beat after the menu closes.
        ctx
          .pruneContext?.()
          .then((result) => {
            if (result.prunedTurns === 0) {
              toast({
                title: "Prune Context",
                description: "Already within the trained window — nothing to prune.",
              })
              return
            }
            const turns = `${result.prunedTurns} turn${result.prunedTurns === 1 ? "" : "s"}`
            toast({
              title: "Prune Context",
              description:
                `Cut ${turns} (~${result.prunedTokens.toLocaleString("en-US")} tokens). ` +
                "The multiplier still applies to new headroom.",
              type: "success",
            })
          })
          .catch(() => {
            toast.error("Prune Context", "Could not compute a context floor for this chat.")
          })
        return true
      }
      if (command.name === "help") {
        const cmdList = SLASH_COMMANDS.map((c) => `/${c.name}`).join(" · ")
        toast({
          title: "Slash Commands",
          description: `${cmdList}`,
        })
        return true
      }
      return true
    }
  }
}
