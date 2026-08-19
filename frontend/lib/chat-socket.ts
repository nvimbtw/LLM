import { WS_URL, getToken, handleSessionExpired } from "./api"
import type { ThinkBudget } from "@/components/chat/budget-selector"

// The backend's exact error frame for a bad/expired token (api/src/ws.rs).
const AUTH_FAILED_MESSAGE = "authentication failed"

// Streams an assistant reply over one socket, per chat.
//
//   client → { token }                            (first frame, auth)
//   server → { type: "ready" }
//   client → { model_name, text, system?, budget?, temperature?, top_k?, top_p? }
//   server → { type: "token", token }             (decoded text, repeated)
//   server → { type: "tool_call", tool, argument } (the reply asked for something)
//   server → { type: "done", message_id, content, usage }
//   server → { type: "tool_result", tool, argument, body }
//   server → { type: "error", error }             (at any point)
//
// A turn that calls a tool runs more than one round, so `token` and `done` can
// each arrive twice — the reply that called, then the reply that answers with
// what came back. The two tool frames are not rendered yet.
//
// The chat exists before the socket opens — its id is in the URL — so there is no
// "which conversation is this" frame and no id to wait for on the way back. The
// reply is written when the last token lands, so a dropped connection loses it
// rather than storing half of one.

/**
 * The decoding knobs a turn is sent with, in the settings' own camelCase. Named
 * as one object rather than three more positional arguments to `streamChat`,
 * which already takes five.
 */
export interface SamplingSettings {
  temperature: number
  topK: number
  topP: number
}

/** One search the reply asked for, reported the moment its call closed. */
export interface ToolCallFrame {
  tool: string
  argument: string
}

/** What that search came back with, once the backend has run it. */
export interface ToolResultFrame extends ToolCallFrame {
  /** The result in the marker vocabulary — a `<|list|>`, or a `<|warn|>`. */
  body: string
}

export interface ChatStreamHandlers {
  onToken: (value: string) => void
  onToolCall: (call: ToolCallFrame) => void
  onToolResult: (result: ToolResultFrame) => void
  /** Fires once per round. `final` is the one that ends the turn. */
  onDone: (info: {
    messageId: string | null
    final: boolean
    /** This round's generation time. A turn's total is the sum of its rounds. */
    elapsedSecs: number | null
  }) => void
  onError: (message: string) => void
}

/** Opens the stream and returns a cancel function. */
export function streamChat(
  chatId: string,
  content: string,
  modelId: string,
  system: string | null,
  // How long the model should reason. "auto" is sent as nothing at all — the
  // engine renders no marker, which is the prompt shape it has always used.
  budget: ThinkBudget,
  sampling: SamplingSettings,
  handlers: ChatStreamHandlers
): () => void {
  const ws = new WebSocket(`${WS_URL}/ws/chats/${chatId}`)
  let settled = false

  const settle = (fn: () => void) => {
    if (settled) return
    settled = true
    fn()
    try {
      ws.close()
    } catch {
      // already closing
    }
  }

  ws.onopen = () => {
    ws.send(JSON.stringify({ token: getToken() }))
  }

  ws.onmessage = (event) => {
    let msg: {
      type?: string
      token?: string
      message_id?: string | null
      error?: string
      tool?: string
      argument?: string
      body?: string
      /** On a `done` frame: whether it ends the turn or only this round. */
      final?: boolean
      usage?: { elapsed_secs?: number }
    }
    try {
      msg = JSON.parse(event.data)
    } catch {
      return
    }

    switch (msg.type) {
      case "ready": {
        // Omit `system` unless it's a non-empty prompt, so the backend treats a
        // blank setting as "no system prompt" rather than an empty turn.
        const trimmedSystem = system?.trim()
        ws.send(
          JSON.stringify({
            model_name: modelId,
            text: content,
            ...(trimmedSystem ? { system: trimmedSystem } : {}),
            // Same rule for the budget: "auto" is the absence of one, so it is
            // omitted rather than sent as a value the engine has to interpret.
            ...(budget !== "auto" ? { budget } : {}),
            // Each knob is sent only when it is a real number. A NaN from a
            // corrupted localStorage entry would serialize to `null` and be
            // rejected by the route, so an unusable value becomes "use the
            // server default" rather than a failed turn.
            ...(Number.isFinite(sampling.temperature)
              ? { temperature: sampling.temperature }
              : {}),
            ...(Number.isFinite(sampling.topK)
              ? { top_k: Math.round(sampling.topK) }
              : {}),
            ...(Number.isFinite(sampling.topP) ? { top_p: sampling.topP } : {}),
          })
        )
        break
      }
      case "token":
        handlers.onToken(msg.token ?? "")
        break
      case "tool_call":
        handlers.onToolCall({
          tool: msg.tool ?? "",
          argument: msg.argument ?? "",
        })
        break
      case "tool_result":
        handlers.onToolResult({
          tool: msg.tool ?? "",
          argument: msg.argument ?? "",
          body: msg.body ?? "",
        })
        break
      case "done": {
        // A turn that called a tool runs more than one round, and the engine
        // ends each of them with `done` — only the one the backend marks final
        // ends the turn. Settling on the first would close the socket before
        // the answer the search was for had been generated.
        //
        // Absent is treated as final: a backend that predates the flag only
        // ever sent one `done`, and hanging is a worse failure than closing.
        const info = {
          messageId: msg.message_id == null ? null : String(msg.message_id),
          elapsedSecs:
            typeof msg.usage?.elapsed_secs === "number"
              ? msg.usage.elapsed_secs
              : null,
        }
        if (msg.final === false) {
          handlers.onDone({ ...info, final: false })
          break
        }
        settle(() => handlers.onDone({ ...info, final: true }))
        break
      }
      case "error":
        if (msg.error === AUTH_FAILED_MESSAGE) {
          settle(() => handleSessionExpired())
          break
        }
        settle(() => handlers.onError(msg.error || "Something went wrong."))
        break
    }
  }

  ws.onerror = () => settle(() => handlers.onError("Connection error."))
  ws.onclose = () => settle(() => handlers.onError("Connection closed."))

  // Closing is the cancel: the reply is written when the last token lands, so a
  // turn nobody is listening to writes nothing.
  return () => settle(() => {})
}
