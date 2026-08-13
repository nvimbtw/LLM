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
//   server → { type: "done", message_id, content, usage }
//   server → { type: "error", error }             (at any point)
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

export interface ChatStreamHandlers {
  onToken: (value: string) => void
  onDone: (info: { messageId: string | null }) => void
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
      case "done":
        settle(() =>
          handlers.onDone({
            messageId: msg.message_id == null ? null : String(msg.message_id),
          })
        )
        break
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
