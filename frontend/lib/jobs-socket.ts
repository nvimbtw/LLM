import { WS_URL, getToken } from "./api"

// One job's live frames, over one socket:
//
//   client → { token }                      (first frame, auth)
//   server → { type: "ready" }
//   server → { type: "state", ... }         (a queued or finished row; then EOF)
//   server → { type: "progress", ... }      (that kind's own payload)
//   server → { type: "done" | "error" | "cancelled", ... }
//   client → { type: "cancel" }
//
// The frames are the LLM engine's own, relayed: the backend flattens each event's
// name onto `type` and passes the payload through. A job that is not on the
// device has no frames — the stream is the row and then end of stream, which is
// what `onInactive` reports.
//
// The board has no socket. It changes rarely and a poll says the same thing as a
// nudge would, without a second live connection to keep alive across sleeps,
// restarts and blips.

export interface JobFrameHandlers {
  // Every frame for the attached job, already parsed. Kept generic because the
  // payload depends on the job's kind, and each panel narrows it itself.
  onFrame: (frame: Record<string, unknown>) => void
  // The job isn't running — it is queued, or already finished. There is nothing
  // to stream and the REST row is the whole story.
  onInactive?: () => void
  onError?: (message: string) => void
  onClose?: () => void
}

export interface Attachment {
  cancel: () => void
  detach: () => void
}

/** Follows one job using its captured bearer token. Terminal frames close observation; only explicit cancel sends a control frame, and detach suppresses all late callbacks. */
export function attachJob(
  idJob: string,
  handlers: JobFrameHandlers,
  token: string | null = getToken()
): Attachment {
  let closed = false
  const ws = new WebSocket(`${WS_URL}/ws/runs/${idJob}`)

  ws.onopen = () => {
    if (!closed) ws.send(JSON.stringify({ token }))
  }

  ws.onmessage = (event) => {
    if (closed) return
    let msg: Record<string, unknown>
    try {
      const value: unknown = JSON.parse(event.data)
      if (!value || typeof value !== "object" || Array.isArray(value)) return
      msg = value as Record<string, unknown>
    } catch {
      return
    }
    switch (msg.type) {
      case "ready":
        return
      case "state":
        handlers.onInactive?.()
        return
      case "error":
        handlers.onError?.(
          typeof msg.error === "string" ? msg.error : "Something went wrong."
        )
        break
    }
    handlers.onFrame(msg)
    if (
      msg.type === "done" ||
      msg.type === "error" ||
      msg.type === "cancelled"
    ) {
      closed = true
      try {
        ws.close()
      } catch {
        return
      }
    }
  }

  ws.onerror = () => {
    if (!closed) handlers.onError?.("Connection error.")
  }
  ws.onclose = () => {
    if (!closed) handlers.onClose?.()
    closed = true
  }

  return {
    /** Requests cooperative cancellation only while this attachment is still live; REST controls remain available after disconnect. */
    cancel: () => {
      if (closed) return
      try {
        ws.send(JSON.stringify({ type: "cancel" }))
      } catch {
        return
      }
    },
    /** Releases observation and suppresses late frames without cancelling durable execution. */
    detach: () => {
      closed = true
      try {
        ws.close()
      } catch {
        return
      }
    },
  }
}
