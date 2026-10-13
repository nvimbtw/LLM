import type { ChatMessage } from "./chat-messages"
import { joinWorking } from "./chat-messages"
import type { ToolResultFrame } from "./chat-socket"

export interface ChatTurn {
  message: ChatMessage
  phase: "answer" | "working"
  insideThinking: boolean
  totalTokens: number
  thinkingTokens: number
  elapsedSecs: number
}

export type ChatTurnEvent =
  | { type: "token"; value: string; tokenId?: number }
  | { type: "tool-call" }
  | { type: "tool-result"; result: ToolResultFrame }
  | { type: "interjected"; text: string }
  | { type: "round-done"; elapsedSecs: number | null }

/** Creates an empty assistant turn with independent token and elapsed-time tallies. */
export function createChatTurn(): ChatTurn {
  return {
    message: { role: "assistant", content: "" },
    phase: "answer",
    insideThinking: false,
    totalTokens: 0,
    thinkingTokens: 0,
    elapsedSecs: 0,
  }
}

/**
 * Appends a decoded delta without inserting whitespace and counts its arrival once.
 * Thinking markers count as thinking tokens; tokens continuing a tool call stay in working.
 */
function appendToken(turn: ChatTurn, value: string): ChatTurn {
  const totalTokens = turn.totalTokens + 1
  if (turn.phase === "working") {
    return {
      ...turn,
      totalTokens,
      message: {
        ...turn.message,
        working: (turn.message.working ?? "") + value,
        tokenCount: totalTokens,
      },
    }
  }
  const thinking = turn.insideThinking || value.includes("<|thinking|>")
  const thinkingTokens = turn.thinkingTokens + Number(thinking)
  return {
    ...turn,
    totalTokens,
    thinkingTokens,
    insideThinking: thinking && !value.includes("<|/thinking|>"),
    message: {
      ...turn.message,
      content: turn.message.content + value,
      tokenCount: totalTokens,
      thinkingTokenCount: thinkingTokens,
    },
  }
}

/**
 * Folds a typed stream event into a new turn, leaving the previous value untouched.
 * Each tool round retires its answer into working; interjections retain their position among answers.
 */
export function reduceChatTurn(turn: ChatTurn, event: ChatTurnEvent): ChatTurn {
  switch (event.type) {
    case "token":
      return appendToken(turn, event.value)
    case "tool-call":
      return turn.phase === "working"
        ? turn
        : {
            ...turn,
            phase: "working",
            message: {
              ...turn.message,
              working: joinWorking(turn.message.working, turn.message.content),
              content: "",
            },
          }
    case "tool-result":
      return {
        ...turn,
        phase: "answer",
        message: {
          ...turn.message,
          toolAnswers: [...(turn.message.toolAnswers ?? []), event.result],
        },
      }
    case "interjected":
      return {
        ...turn,
        message: {
          ...turn.message,
          interjections: [
            ...(turn.message.interjections ?? []),
            {
              text: event.text,
              afterAnswers: turn.message.toolAnswers?.length ?? 0,
            },
          ],
        },
      }
    case "round-done":
      return {
        ...turn,
        elapsedSecs: turn.elapsedSecs + (event.elapsedSecs ?? 0),
      }
  }
}

/** Adds live-only tallies to committed history without changing its persisted content or attribution. */
export function reconcileChatTurn(
  messages: ChatMessage[],
  turn: ChatTurn
): ChatMessage[] {
  const last = messages.at(-1)
  if (last?.role !== "assistant") return messages
  return [
    ...messages.slice(0, -1),
    {
      ...last,
      tokenCount: turn.totalTokens,
      thinkingTokenCount: turn.thinkingTokens,
      elapsedSecs: turn.elapsedSecs || undefined,
    },
  ]
}
