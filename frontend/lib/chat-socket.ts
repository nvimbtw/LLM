import { WS_URL, getToken, handleSessionExpired } from "./api"

// The backend's exact error frame for a bad/expired token (routes/chat.rs).
const AUTH_FAILED_MESSAGE = "authentication failed"

// Streams an assistant reply over the /chat WebSocket.
//
// Wire protocol:
//   client → { token }                            (first frame, auth)
//   server → { type: "ready" }
//   client → { chat_id, content, model_id, system }  (chat_id null = new chat;
//                                                      system optional/omitted)
//   server → { type: "token", value }             (decoded token text, repeated)
//   server → { type: "done", chat_id, id_message }
//   server → { type: "error", message }           (at any point)

export interface ChatStreamHandlers {
  onToken: (value: string) => void
  onDone: (info: { chatId: string; messageId: string }) => void
  onError: (message: string) => void
}

/** Opens the stream and returns a cancel function. */
export function streamChat(
  chatId: string | null,
  content: string,
  modelId: string | null,
  system: string | null,
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
    let msg: { type?: string; value?: string; chat_id?: string; id_message?: string; message?: string }
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
          })
        )
        break
      }
      case "token":
        handlers.onToken(msg.value ?? "")
        break
      case "done":
        settle(() =>
          handlers.onDone({
            chatId: String(msg.chat_id),
            messageId: String(msg.id_message),
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

  return () => settle(() => {})
}
