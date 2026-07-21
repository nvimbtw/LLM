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

// Which kind of run: pretrain a fresh model, or supervised-fine-tune (SFT) an
// existing one on chat data. Matches the backend `TrainMode` (serialized
// lowercase). Absent on older clients → the backend defaults to "pretrain".
export type TrainMode = "pretrain" | "sft"

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
  // How many uploaded books to train on, picked at random. 0 = use all.
  num_books: number
  // How many synthetic documents to mix in, picked at random. 0 = none.
  num_synthetic: number
  // Pretrain (default) vs SFT fine-tune.
  mode: TrainMode
  // For SFT only: the id of the base model to fine-tune from. Omitted for
  // pretrain; the backend rejects an SFT request without a resolvable one.
  base_model_id?: string
}

export type JobStatus =
  | "tokenizing"
  | "training"
  | "done"
  | "cancelled"
  | "failed"

export interface TokenizeStats {
  // Pretrain phase-1 stats (BPE tokenizer + corpus build).
  files: number
  chars: number
  vocab: number
  merges: number
  // SFT phase-1 stats (chat dataset build). The backend reuses the `tokenize`
  // frame for SFT's "build dataset" phase but emits these fields instead of the
  // pretrain ones; their presence is how the panel tells the two modes apart.
  conversations: number
  total_tokens: number
  trainable_tokens: number
  progress: number
  done: boolean
}

// One report's worth of training diagnostics (grad norms, update:weight ratios,
// activation RMS, attention entropy, embedding anisotropy). Emitted on mid-epoch
// report frames only, tagged with the same global `step` as the loss point.
// `*_layers` arrays are indexed by transformer layer (0 = first block);
// `attn_entropy[layer]` holds one normalized entropy per attention head.
export interface DiagPoint {
  step: number
  grad_global: number
  grad_layers: number[]
  grad_embedding: number
  grad_unembed: number
  update_ratio_layers: number[]
  act_rms_layers: number[]
  attn_entropy: number[][]
  emb_cos_mean: number
  emb_cos_p5: number
  emb_cos_p95: number
  // Z-scored activation-distribution histogram (fixed bin count, fractions summing
  // to ~1) of the deepest block's residual stream — one per report.
  act_hist: number[]
  // 2-D PCA projection of the frequent tokens' embeddings ([x, y] per token).
  emb_pca: [number, number][]
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
  // Spread of per-chunk loss within just this report window.
  batch_min: number
  batch_avg: number
  batch_max: number
  eta_secs: number
  epoch_done: boolean
  epoch_duration_secs: number
  checkpoint_path: string | null
  step: number
  // Diagnostics sampled on this report frame, or null on epoch-boundary frames.
  diag: DiagPoint | null
}

export interface TrainingSnapshot {
  status: JobStatus
  config: TrainConfig
  tokenize: Partial<TokenizeStats>
  train: Partial<TrainStats>
  loss_series: [number, number][]
  batch_series: [number, number, number, number][]
  diag_series: DiagPoint[]
  log: string[]
  error: string | null
  // Present (non-null) only when a *successful* run registered a model — the
  // backend fills these on the `done` and terminal `snapshot` frames so the
  // client can seed a follow-up SFT run with this model as its base without
  // hunting for it in the model list ("Continue to SFT"). Null for in-flight,
  // cancelled, or failed runs.
  id_model?: string | null
  model_name?: string | null
}

// The identifiers of a model a completed run produced. Absent when a run
// cancelled/failed or hasn't finished.
export interface FinishedModel {
  id: string
  name: string
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
  onDone: (info: {
    status: string
    model_dir: string
    id_model?: string | null
    model_name?: string | null
  }) => void
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
          id_model:
            typeof msg.id_model === "string" ? msg.id_model : null,
          model_name:
            typeof msg.model_name === "string" ? msg.model_name : null,
        })
        break
      case "error":
        handlers.onError(
          typeof msg.message === "string"
            ? msg.message
            : "Something went wrong."
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
