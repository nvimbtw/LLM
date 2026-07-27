import { WS_URL, getToken } from "./api"

// The queue's live channel, generalized from what used to be `training-socket.ts`.
//
// One socket, two subscriptions:
//
//   client → { token }                       (first frame, auth)
//   server → { type: "ready" }
//
//   client → { subscribe: "queue" }          (follow the board)
//   server → { type: "changed" }             (something moved; refetch)
//   server → { type: "started", id_job }     (a job reached the GPU)
//   server → { type: "progress", id_job, progress, stage }
//
//   client → { job_id }                      (follow one job)
//   server → { type: "snapshot", ... }       (catch-up on attach)
//   server → { type: "inactive" }            (not the job on the GPU)
//   server → ...that kind's frames...        (tokenize/log/train, progress, token)
//   server → { type: "done" | "error", ... }
//   client → { action: "cancel" }
//
// Board events are deliberately nudges rather than payloads: the client refetches
// `GET /jobs`, so there is exactly one description of a job rather than a REST
// shape and a socket shape that can drift apart.
//
// `progress` is the one exception, and carries its values. The nudge shape works
// because the board changes rarely; progress moves continuously, and refetching
// the whole board every couple of seconds to advance two scalars on one row is the
// wrong trade. It patches a row the client already has rather than describing a
// job afresh, so the single-description rule still holds.

export interface BoardHandlers {
  onChanged: () => void
  onStarted?: (idJob: string) => void
  onProgress?: (idJob: string, progress: number, stage: string | null) => void
}

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

function connect(
  subscribe: Record<string, unknown>,
  onMessage: (msg: Record<string, unknown>) => void
) {
  const ws = new WebSocket(`${WS_URL}/jobs/ws`)

  ws.onopen = () => {
    ws.send(JSON.stringify({ token: getToken() }))
  }

  ws.onmessage = (event) => {
    let msg: Record<string, unknown>
    try {
      msg = JSON.parse(event.data) as Record<string, unknown>
    } catch {
      return
    }
    if (msg.type === "ready") {
      ws.send(JSON.stringify(subscribe))
      return
    }
    onMessage(msg)
  }

  return ws
}

// Reconnect backoff for the board socket. Starts fast so a blip is invisible, and
// tops out low enough that a backend restart is picked up within seconds.
const RETRY_MIN_MS = 1000
const RETRY_MAX_MS = 15000

/**
 * Follows the whole board. The handler refetches; this only says when.
 *
 * Reconnects on its own. This socket is opened once by the provider and expected
 * to live for the whole session, so a drop it didn't recover from — a backend
 * restart, a sleep, a blip — would freeze the queue everywhere it is shown until
 * the page was reloaded, with nothing on screen saying it had gone stale. Each
 * successful reconnect refetches, since anything could have happened while it was
 * away.
 */
export function watchBoard(handlers: BoardHandlers): () => void {
  let socket: WebSocket | null = null
  let retryMs = RETRY_MIN_MS
  let timer: ReturnType<typeof setTimeout> | null = null
  let stopped = false

  const open = () => {
    socket = connectBoard(
      handlers,
      // Connected: forget the backoff, so the next unrelated blip is recovered
      // from as quickly as the first rather than inheriting an old outage's delay.
      () => {
        retryMs = RETRY_MIN_MS
      },
      () => {
        if (stopped) return
        timer = setTimeout(open, retryMs)
        // Back off up to the ceiling, so a backend that stays down isn't hammered.
        retryMs = Math.min(retryMs * 2, RETRY_MAX_MS)
      }
    )
  }

  open()

  return () => {
    stopped = true
    if (timer) clearTimeout(timer)
    try {
      socket?.close()
    } catch {
      // already closing
    }
  }
}

function connectBoard(
  handlers: BoardHandlers,
  onOpen: () => void,
  onLost: () => void
): WebSocket {
  let lost = false
  // Both `onerror` and `onclose` fire for a single failure; only the first should
  // schedule a retry, or the backoff doubles twice per drop.
  const lose = () => {
    if (lost) return
    lost = true
    onLost()
  }

  const ws = connect({ subscribe: "queue" }, (msg) => {
    // Progress patches a row in place; it must not trigger a refetch, or the
    // whole point of carrying the payload is lost.
    if (msg.type === "progress") {
      if (typeof msg.id_job === "string" && typeof msg.progress === "number") {
        handlers.onProgress?.(
          msg.id_job,
          msg.progress,
          typeof msg.stage === "string" ? msg.stage : null
        )
      }
      return
    }
    if (msg.type === "started" && typeof msg.id_job === "string") {
      handlers.onStarted?.(msg.id_job)
    }
    // A `started` event also changes the board, so both kinds nudge a refetch.
    handlers.onChanged()
  })

  // A fresh connection has missed everything that happened while it was down, so
  // it resyncs rather than waiting for the next event to tell it something moved.
  ws.addEventListener("open", () => {
    onOpen()
    handlers.onChanged()
  })
  ws.addEventListener("close", lose)
  ws.addEventListener("error", lose)

  return ws
}

/** Follows one job's live frames. Returns cancel + detach controls. */
export function attachJob(
  idJob: string,
  handlers: JobFrameHandlers
): Attachment {
  let closed = false

  const ws = connect({ job_id: idJob }, (msg) => {
    if (msg.type === "inactive") {
      handlers.onInactive?.()
      return
    }
    if (msg.type === "error") {
      handlers.onError?.(
        typeof msg.message === "string" ? msg.message : "Something went wrong."
      )
    }
    handlers.onFrame(msg)
  })

  ws.onerror = () => {
    // A socket that we closed on purpose also fires this; only a live one is a
    // real failure worth reporting.
    if (!closed) handlers.onError?.("Connection error.")
  }

  return {
    cancel: () => {
      try {
        ws.send(JSON.stringify({ action: "cancel" }))
      } catch {
        // socket not open; the REST cancel is the fallback
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
