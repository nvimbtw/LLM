import { API_URL, WS_URL, authHeaders, getToken } from "./api"

// REST starts a run; the WebSocket only *attaches* to it. That split is what
// makes a run survive a tab close: the job keeps going on the backend and a
// fresh socket re-attaches by id.
//
// Socket wire protocol:
//   client → { token }                          (first frame, auth)
//   server → { type: "ready" }
//   client → { job_id }                         (attach to a run)
//   server → { type: "snapshot", ... }          (catch-up on attach)
//   server → { type: "tokenize", ... }          (phase 1 stat updates)
//   server → { type: "log", line }              (raw tokenizer log line)
//   server → { type: "train", ... }             (phase 2 progress)
//   server → { type: "done", status, model_dir }
//   server → { type: "error", message }
//   client → { action: "cancel" }               (stop the run)

export interface TrainConfig {
  model_name: string
  dimensions: number
  ctx_window: number
  heads: number
  layers: number
  epochs: number
  lr: number
  ff_multiplier: number
  max_vocab: number
}

export type JobStatus =
  | "tokenizing"
  | "training"
  | "done"
  | "cancelled"
  | "failed"

export interface TokenizeStats {
  files: number
  chars: number
  vocab: number
  merges: number
  progress: number
  done: boolean
}

export interface TrainStats {
  epoch: number
  total_epochs: number
  chunk: number
  total_chunks: number
  loss: number
  avg_loss: number
  min_loss: number
  max_loss: number
  eta_secs: number
  epoch_done: boolean
  epoch_duration_secs: number
  checkpoint_path: string | null
  step: number
}

export interface TrainingSnapshot {
  status: JobStatus
  config: TrainConfig
  tokenize: Partial<TokenizeStats>
  train: Partial<TrainStats>
  loss_series: [number, number][]
  log: string[]
  error: string | null
}

export interface JobSummary {
  job_id: string
  status: JobStatus
  model_name: string
}

export interface TrainingHandlers {
  onSnapshot: (snap: TrainingSnapshot) => void
  onTokenize: (stats: TokenizeStats) => void
  onLog: (line: string) => void
  onTrain: (stats: TrainStats) => void
  onDone: (info: { status: string; model_dir: string }) => void
  onError: (message: string) => void
}

export interface Attachment {
  cancel: () => void
  detach: () => void
}

export async function startTraining(config: TrainConfig): Promise<string> {
  const res = await fetch(`${API_URL}/training`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify(config),
  })
  if (!res.ok) {
    const message = await res.text()
    throw new Error(message || "Failed to start training.")
  }
  const data = (await res.json()) as { job_id: string }
  return data.job_id
}

export async function listJobs(): Promise<JobSummary[]> {
  const res = await fetch(`${API_URL}/training`, { headers: authHeaders() })
  if (!res.ok) return []
  return (await res.json()) as JobSummary[]
}

/** Attaches to a run's progress stream. Returns cancel + detach controls. */
export function attachTraining(
  jobId: string,
  handlers: TrainingHandlers
): Attachment {
  const ws = new WebSocket(`${WS_URL}/training/ws`)
  let closed = false

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

    switch (msg.type) {
      case "ready":
        ws.send(JSON.stringify({ job_id: jobId }))
        break
      case "snapshot":
        handlers.onSnapshot(msg as unknown as TrainingSnapshot)
        break
      case "tokenize":
        handlers.onTokenize(msg as unknown as TokenizeStats)
        break
      case "log":
        handlers.onLog(typeof msg.line === "string" ? msg.line : "")
        break
      case "train":
        handlers.onTrain(msg as unknown as TrainStats)
        break
      case "done":
        handlers.onDone({
          status: String(msg.status ?? ""),
          model_dir: String(msg.model_dir ?? ""),
        })
        break
      case "error":
        handlers.onError(
          typeof msg.message === "string" ? msg.message : "Something went wrong."
        )
        break
    }
  }

  ws.onerror = () => {
    if (!closed) handlers.onError("Connection error.")
  }

  return {
    cancel: () => {
      try {
        ws.send(JSON.stringify({ action: "cancel" }))
      } catch {
        // socket not open; nothing to cancel
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
