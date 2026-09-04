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
// The frames are the engine's own, relayed: the backend flattens each event's
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
}

export interface Attachment {
  cancel: () => void
  detach: () => void
}

/** Follows one job's live frames. Returns cancel + detach controls. */
export function attachJob(
  idJob: string,
  handlers: JobFrameHandlers
): Attachment {
  let closed = false
  const ws = new WebSocket(`${WS_URL}/ws/runs/${idJob}`)

  ws.onopen = () => ws.send(JSON.stringify({ token: getToken() }))

  ws.onmessage = (event) => {
    let msg: Record<string, unknown>
    try {
      msg = JSON.parse(event.data) as Record<string, unknown>
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
    // A terminal frame ends the run's stream for good, same as `final` does on the
    // chat socket. Closing here — rather than only on unmount/detach — means a job
    // page left open after its run finishes stops listening immediately instead of
    // staying attached (and able to keep writing state) for as long as the page
    // happens to stay mounted.
    if (msg.type === "done" || msg.type === "error" || msg.type === "cancelled") {
      closed = true
      try {
        ws.close()
      } catch {
        // already closing
      }
    }
  }

  ws.onerror = () => {
    // A socket we closed on purpose also fires this; only a live one is a real
    // failure worth reporting.
    if (!closed) handlers.onError?.("Connection error.")
  }

  return {
    cancel: () => {
      try {
        ws.send(JSON.stringify({ type: "cancel" }))
      } catch {
        // Socket not open; the REST cancel is the fallback.
      }
    },
    detach: () => {
      closed = true
      try {
        ws.close()
      } catch {
        // already closing
      }
    },
  }
}
