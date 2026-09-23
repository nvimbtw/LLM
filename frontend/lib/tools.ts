/**
 * What each tool is called in the interface.
 *
 * A tool is a row in `llm-engine/llm/chat/tools.py` and a name on the wire — the
 * vocabulary stopped having a marker per tool (see
 * `docs/decisions/generic-tool-calls.md`), so `web-search` and `bash` arrive
 * here as ordinary strings and nothing upstream knows how to say them in
 * English. This table is that, and it is the only place it is written down:
 * the thread's status row says what is running right now, the trace says what
 * ran, and the two must not disagree about which tool it was.
 *
 * A tool with no entry falls back to its own wire name. That is deliberately
 * usable rather than a placeholder — adding a tool must not require touching
 * the frontend before it works, only before it reads well.
 */
export interface ToolWords {
  /** For a label naming the tool itself: "Search: …". */
  noun: string
  /** Mid-flight, present tense. */
  running: string
  /** Afterwards, past tense — a trace is read once it has happened. */
  done: string
  failed?: string
}

const TOOL_WORDS: Record<string, ToolWords> = {
  "web-search": { noun: "Search", running: "Searching", done: "Searched" },
  bash: { noun: "Bash", running: "Running", done: "Ran" },
  todo: {
    noun: "Plan",
    running: "Updating plan",
    done: "Updated plan",
    failed: "Plan unchanged",
  },
}

/**
 * A call whose JSON the model mangled has no tool name at all (`callOf`
 * returns an empty one). It is still shown, because the model did try — so it
 * needs words that claim nothing about which tool it was.
 */
const UNKNOWN: ToolWords = { noun: "Tool", running: "Calling", done: "Called" }

export function toolWords(tool: string): ToolWords {
  if (!tool) return UNKNOWN
  return TOOL_WORDS[tool] ?? { noun: tool, running: tool, done: tool }
}

export function toolArgument(tool: string, argument: string): string {
  if (tool !== "todo") return argument
  try {
    const value: unknown = JSON.parse(argument)
    if (!value || typeof value !== "object") return argument
    const fields = value as Record<string, unknown>
    const subject =
      typeof fields.title === "string"
        ? fields.title
        : typeof fields.id === "string"
          ? fields.id
          : "tasks"
    switch (fields.action) {
      case "create":
        return `Create plan: ${subject}`
      case "add":
        return `Add ${subject}`
      case "update":
        return `Update ${subject}`
      case "complete":
        return `Complete ${subject}`
      case "list":
        return "Read the current plan"
      default:
        return argument
    }
  } catch {
    return argument
  }
}
