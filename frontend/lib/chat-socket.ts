import { WS_URL, getToken, handleSessionExpired } from "./api"
import type { ThinkBudget } from "@/lib/think-budget"

// The backend's exact error frame for a bad/expired token (api/src/ws.rs).
const AUTH_FAILED_MESSAGE = "authentication failed"

// Streams an assistant reply over one socket, per chat.
//
//   client → { token }                            (first frame, auth)
//   server → { type: "ready" }
//   client → { model_name, text, attachment_ids?, system?, budget?, temperature?,
//              top_k?, top_p?, context_multiplier?, history_floor? }
//   server → { type: "token", token }             (decoded text, repeated)
//   server → { type: "tool_call", tool, argument } (the reply asked for something)
//   server → { type: "done", message_id, content, usage }
//   server → { type: "tool_result", tool, argument, body }
//   client → { type: "interject", text }          (any time after the above)
//   server → { type: "interjected", text }        (spliced into the next round)
//   server → { type: "error", error }             (at any point)
//
// A turn that calls a tool runs more than one round, so `token` and `done` can
// each arrive twice — the reply that called, then the reply that answers with
// what came back. The two tool frames are not rendered yet.
//
// `interject` is the client's own addition to a turn already running: sent any
// time after the opening question, over the same socket, while the composer
// stays open for it (see `hooks/use-chat.ts`). The backend can only splice it
// in between a round's tool result and the next round's request — the one
// window where the next thing the model reads hasn't been decided yet — so it
// is buffered until then and confirmed with `interjected` the moment it lands,
// which is also the frontend's cue to move it out of the queued list and into
// the turn's trace. A message that never gets picked up (the turn ends with no
// further rounds) is not confirmed; the frontend sends it as an ordinary new
// turn once the socket closes rather than losing it.
//
// `attachment_ids` names files already stored (see lib/attachments.ts). They are
// bound to the message this turn writes and go no further: nothing about them
// reaches the prompt, which is why they need no room in the context window.
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
  contextMultiplier: number
}

/**
 * The turn being asked. An object rather than two more positional arguments to
 * `streamChat`, which already takes as many as it can carry — and the two
 * belong together anyway: attachments are part of the question, and are dropped
 * with it on the follow-up round a tool call forces.
 */
export interface Question {
  text: string
  /** Ids from `uploadAttachment`, in the order they were added. */
  attachmentIds: string[]
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
  /** A queued `interject` actually made it into the next round. */
  onInterjected: (text: string) => void
  onError: (message: string) => void
}

/** What driving a turn hands back: how to stop it, and how to steer it. */
export interface ChatStreamHandle {
  /** Closing is the cancel: the reply is written when the last token lands, so
   * a turn nobody is listening to writes nothing. */
  cancel: () => void
  /**
   * Send a message over this turn's own socket rather than opening a new one.
   * A no-op past the turn's end (the socket is already closing or closed) —
   * the caller is expected to fall back to an ordinary new turn in that case,
   * since there is no `interjected` coming back to confirm this one.
   */
  interject: (text: string) => void
}

/** Opens the stream and returns a handle to steer or stop it. */
export function streamChat(
  chatId: string,
  question: Question,
  modelId: string,
  system: string | null,
  // How long the model should reason. "auto" is sent as nothing at all — the
  // engine renders no marker, which is the prompt shape it has always used.
  budget: ThinkBudget,
  sampling: SamplingSettings,
  // A message id from `lib/context-floor.ts`, or null. Resent on every turn
  // rather than remembered here — the socket is per-turn, the floor is
  // per-chat, and this is the seam where the two meet.
  historyFloor: string | null,
  handlers: ChatStreamHandlers
): ChatStreamHandle {
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
      /** On an `interjected` frame: the text that was just spliced in. */
      text?: string
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
            text: question.text,
            // Omitted when empty, so a turn with no files is byte-for-byte the
            // request it always was.
            ...(question.attachmentIds.length
              ? { attachment_ids: question.attachmentIds }
              : {}),
            ...(trimmedSystem ? { system: trimmedSystem } : {}),
            // Same rule for the budget: "auto" is the absence of one, so it is
            // omitted rather than sent as a value the LLM engine has to interpret.
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
            ...(Number.isFinite(sampling.contextMultiplier)
              ? { context_multiplier: sampling.contextMultiplier }
              : {}),
            ...(historyFloor ? { history_floor: historyFloor } : {}),
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
      case "interjected":
        handlers.onInterjected(msg.text ?? "")
        break
      case "done": {
        // A turn that called a tool runs more than one round, and the LLM engine
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

  return {
    // Closing is the cancel: the reply is written when the last token lands,
    // so a turn nobody is listening to writes nothing.
    cancel: () => settle(() => {}),
    interject: (text: string) => {
      // Settled means the socket is closing or already gone — nothing sent
      // here would be read, and the caller (`hooks/use-chat.ts`) is what
      // falls back to sending it as a fresh turn once this one's `onDone` or
      // `onError` fires.
      if (settled || ws.readyState !== WebSocket.OPEN) return
      ws.send(JSON.stringify({ type: "interject", text }))
    },
  }
}
