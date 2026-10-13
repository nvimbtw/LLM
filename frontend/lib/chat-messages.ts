import { mapAttachment, type Attachment } from "./attachments"
import type { MessageWire } from "./chat-history"
import { callOf, parseMessage, TOOL_CALL_CLOSE } from "./message-parser"
import type { Feedback } from "./feedback"

export interface ToolAnswer {
  tool: string
  argument: string
  body: string
}

export interface Interjection {
  text: string
  /** Number of tool answers already recorded when the text reached the model. */
  afterAnswers: number
}

export interface ChatMessage {
  role: "user" | "assistant"
  /** The answer. For a turn that used a tool, the *last* round's reply. */
  content: string
  /**
   * The working that preceded the answer — the first round's reply, holding its
   * reasoning and the calls it made. Absent on a turn that never called
   * anything, whose reasoning is inside `content` as it always was.
   */
  working?: string
  /** What each call in `working` came back with, in the order they closed. */
  toolAnswers?: ToolAnswer[]
  /** A person's message actually spliced into this turn, in the order they
   * landed — see the `interjected` frame in `lib/chat-socket.ts`. */
  interjections?: Interjection[]
  /** Seconds the whole turn took, summed across its rounds. */
  elapsedSecs?: number
  id?: string
  feedback?: Feedback
  tokenCount?: number
  thinkingTokenCount?: number
  modelName?: string
  attachments?: Attachment[]
}

/** Joins independent rounds with one boundary space while preserving decoded deltas within each round. */
export function joinWorking(
  working: string | undefined,
  round: string
): string {
  if (working === undefined) return round
  if (!round) return working
  return working ? `${working} ${round}` : round
}

/**
 * Folds chronological stored rows into visible turns without changing the input rows.
 * Tools retire the previous round into working; users between a tool answer and its continuation are interjections.
 * Orphaned tool rows are ignored rather than inventing an assistant turn.
 */
export function groupTurns(rows: MessageWire[]): ChatMessage[] {
  const grouped: ChatMessage[] = []

  for (const row of rows) {
    const previous = grouped[grouped.length - 1]

    if (row.role === "tool") {
      if (previous?.role !== "assistant") continue
      previous.working = joinWorking(previous.working, previous.content)
      previous.toolAnswers = [
        ...(previous.toolAnswers ?? []),
        parseToolRow(row.content),
      ]
      previous.content = ""
      continue
    }

    if (
      row.role === "user" &&
      previous?.role === "assistant" &&
      previous.toolAnswers &&
      !previous.content
    ) {
      previous.interjections = [
        ...(previous.interjections ?? []),
        { text: row.content, afterAnswers: previous.toolAnswers.length },
      ]
      continue
    }

    if (
      row.role === "assistant" &&
      previous?.toolAnswers &&
      !previous.content
    ) {
      previous.content = row.content
      previous.id = row.message_id
      previous.modelName = row.model_name ?? undefined
      previous.feedback = row.feedback ?? undefined
      continue
    }

    grouped.push({
      id: row.message_id,
      role: row.role,
      content: row.content,
      attachments: row.attachments?.length
        ? row.attachments.map(mapAttachment)
        : undefined,
      modelName: row.model_name ?? undefined,
      feedback: row.feedback ?? undefined,
    })
  }

  return grouped
}

/** Extracts the repeated call and result from a stored tool row, retaining unrecognized result text verbatim. */
function parseToolRow(content: string): ToolAnswer {
  for (const node of parseMessage(content).children) {
    const call = callOf(node)
    if (!call) continue
    const at = content.indexOf(TOOL_CALL_CLOSE)
    return {
      ...call,
      body: at === -1 ? content : content.slice(at + TOOL_CALL_CLOSE.length),
    }
  }
  return { tool: "", argument: "", body: content }
}
