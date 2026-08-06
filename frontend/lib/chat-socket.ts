import { WS_URL, getToken, handleSessionExpired } from "./api"
import type { ThinkBudget } from "@/components/chat/budget-selector"

// The backend's exact error frame for a bad/expired token (routes/chat.rs).
const AUTH_FAILED_MESSAGE = "authentication failed"

// Streams an assistant reply over the /chat WebSocket.
//
// Wire protocol:
//   client → { token }                            (first frame, auth)
//   server → { type: "ready" }
//   client → { chat_id, content, model_id, system, budget,
//              temperature, top_k, top_p }
//                                    (chat_id null = new chat; system and budget
//                                     optional/omitted; the three decoding knobs
//                                     are omitted individually when unset, and
//                                     the backend falls back to its own defaults)
//   server → { type: "queued", job_id, chat_id, ahead }
//   server → { type: "token", value }             (decoded token text, repeated)
//   server → { type: "done", chat_id, id_message }
//   server → { type: "error", message }           (at any point)
//   client → { action: "cancel" }                 (drop the turn, queued or running)
//
// The `queued` frame is new: a reply is a job on the GPU queue now, so a turn can
// wait behind other work before its first token. It always arrives — a turn that
// starts immediately reports `ahead: 0` — so the client has one path rather than
// two. `id_message` on `done` is null when a cancelled turn produced nothing.

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
  onQueued: (info: { jobId: string; ahead: number }) => void
  onToken: (value: string) => void
  onDone: (info: { chatId: string; messageId: string | null }) => void
  onError: (message: string) => void
}

/** Opens the stream and returns a cancel function. */
export function streamChat(
  chatId: string | null,
  content: string,
  modelId: string | null,
  system: string | null,
  // How long the model should reason. "auto" is sent as nothing at all — the
  // backend renders no marker, which is the prompt shape it has always used.
  budget: ThinkBudget,
  sampling: SamplingSettings,
  handlers: ChatStreamHandlers
): () => void {
  const ws = new WebSocket(`${WS_URL}/chat/ws`)
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
      value?: string
      chat_id?: string
      id_message?: string | null
      message?: string
      job_id?: string
      ahead?: number
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
            chat_id: chatId,
            content,
            model_id: modelId,
            ...(trimmedSystem ? { system: trimmedSystem } : {}),
            // Same rule for the budget: "auto" is the absence of one, so it is
            // omitted rather than sent as a value the backend has to interpret.
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
      case "queued":
        handlers.onQueued({
          jobId: String(msg.job_id),
          ahead: typeof msg.ahead === "number" ? msg.ahead : 0,
        })
        break
      case "token":
        handlers.onToken(msg.value ?? "")
        break
      case "done":
        settle(() =>
          handlers.onDone({
            chatId: String(msg.chat_id),
            // Null when a cancelled turn produced nothing worth persisting. The
            // caller refetches either way, so this is only a signal, not an id.
            messageId: msg.id_message == null ? null : String(msg.id_message),
          })
        )
        break
      case "error":
        if (msg.message === AUTH_FAILED_MESSAGE) {
          settle(() => handleSessionExpired())
          break
        }
        settle(() => handlers.onError(msg.message || "Something went wrong."))
        break
    }
  }

  ws.onerror = () => settle(() => handlers.onError("Connection error."))
  ws.onclose = () => settle(() => handlers.onError("Connection closed."))

  // Cancelling has to reach the server, not just drop the socket: the turn is a
  // queued job that would otherwise run to completion (and persist its reply)
  // with nobody watching. Sent before the close so it isn't dropped with it.
  return () =>
    settle(() => {
      try {
        ws.send(JSON.stringify({ action: "cancel" }))
      } catch {
        // Socket already gone; the job outlives it either way.
      }
    })
}
