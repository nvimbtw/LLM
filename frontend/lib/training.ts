import { API_URL, authHeaders } from "./api"

// The shape of a training run: its config, and the frames it reports while it
// executes.
//
// Starting and following a run now belong to the queue — a run is a `train` job,
// started through `lib/jobs.ts` and followed through `lib/jobs-socket.ts`. What
// stays here is the vocabulary both of those speak about training specifically,
// plus the dataset list the composer needs.

// Which kind of run: pretrain a fresh model, supervised-fine-tune (SFT) an
// existing one on chat data, preference-tune (KTO) one on thumbs-rated replies, or
// self-train (RAFT) one on its own generations that cleared the catalogue's bar.
//
// RAFT and KTO are the two halves of a post-training round and run in that order:
// RAFT moves the model toward the good region (dense signal, full token
// distribution), then KTO suppresses what stayed bad (one bit per example, but the
// only thing here that can push probability *down*).
// Matches the backend `TrainMode` (serialized lowercase). Absent on older clients →
// the backend defaults to "pretrain".
export type TrainMode = "pretrain" | "sft" | "kto" | "raft"

// The two modes that start from an existing model rather than random weights.
// Both inherit the base's architecture and both require a `base_model_id`.
export function isFineTune(mode: TrainMode): boolean {
  return mode === "sft" || mode === "kto" || mode === "raft"
}

// One picked dataset: the name of an uploads/ subfolder and how many of its .txt
// files to sample at random (0 = all of that dataset).
export interface DatasetSelection {
  name: string
  count: number
}

// A selectable dataset returned by GET /training/datasets: an uploads/ subfolder
// and its total .txt count (the cap the picker shows as "n / files").
export interface DatasetInfo {
  name: string
  files: number
}

// One curriculum stage: a set of datasets trained for a fixed number of epochs
// before the run advances to the next stage. Mirrors the backend `CurriculumStage`.
export interface CurriculumStage {
  name?: string
  datasets: DatasetSelection[]
  epochs: number
}

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
  // Which upload datasets (subfolders of uploads/) to train on, each with how
  // many of its files to sample (0 = all of that dataset). Pretrain only.
  datasets: DatasetSelection[]
  // Ordered curriculum stages (pretrain only). When non-empty, the run trains
  // each stage's datasets for its own epoch count instead of the flat `datasets`
  // + `epochs`. Empty = the plain single-pass behaviour.
  curriculum: CurriculumStage[]
  // Reorder documents each epoch (with an <|endoftext|> separator between them)
  // so the model doesn't see a fixed data order. Pretrain only.
  shuffle: boolean
  // Drop the <|thinking|> blocks (and their budget markers) from the chat data
  // this run trains on, so the model learns to answer directly.
  //
  // A choice for SFT only. Pretrain always strips whatever this says — the
  // backend forces it true at enqueue so the stored config stays truthful — and
  // RAFT/KTO ignore it, since they train on the model's own generations, which
  // carry no thinking block to remove.
  strip_thinking: boolean
  // Pretrain (default), SFT fine-tune, or KTO preference tune.
  mode: TrainMode
  // For the fine-tuning modes: the id of the base model to start from. Omitted for
  // pretrain; the backend rejects a fine-tune request without a resolvable one.
  base_model_id?: string
  // KTO only: strength of the KL anchor holding the tune near the base model.
  // Higher keeps it closer. Ignored by the other two modes.
  beta?: number
}

// Where a run is *within itself*. Deliberately not the same thing as the job's
// `JobStatus` in lib/jobs.ts: that says whether the queue is running this row,
// this says which of the run's two phases it is in. A queued run has no phase.
export type RunPhase =
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
  // Prose tokens mixed in from the base model's pretraining corpus, all trainable.
  // 0 means the tokenizer guard refused replay — the fine-tune has nothing holding
  // it to prose, which is catastrophic forgetting with no warning attached.
  replay_tokens: number
  // Conversations refused for rendering past the base model's context window.
  // Small is expected; large means the SFT set was prepared for a different window.
  skipped_too_long: number
  // KTO's phase 1 counts rated *examples* rather than conversations, and reports
  // the split it found. Same frame, one more name for "how many".
  examples: number
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
  // 3-D PCA projection of the same tokens ([x, y, z] per token, same principal axes
  // as `emb_pca`) — the vertices of the rotating embedding globe.
  emb_pca3: [number, number, number][]
}

export interface TrainStats {
  epoch: number
  total_epochs: number
  chunk: number
  total_chunks: number
  // Curriculum stage this frame belongs to (0-based) and its optional label, so
  // the chart can mark stage transitions. A non-curriculum run is a single stage 0.
  stage: number
  stage_name: string | null
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
  status: RunPhase
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

export async function listDatasets(): Promise<DatasetInfo[]> {
  const res = await fetch(`${API_URL}/training/datasets`, {
    headers: authHeaders(),
  })
  if (!res.ok) return []
  return (await res.json()) as DatasetInfo[]
}
