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
// documents to take (0 = all of that dataset).
//
// `id` is the data engine's — the engine stores no corpora and selects by id,
// because a name is something a person can change without changing a token.
// `name` rides along so a stored config still reads as something when the
// dataset list is not to hand; the data engine's own name wins at run time.
export interface DatasetSelection {
  id: string
  name?: string
  count: number
}

// A selectable dataset, as the picker needs it. Narrowed from the data engine's
// `Dataset` (see `lib/data.ts`) so the rail depends on the three fields it
// actually renders rather than on the whole row.
export interface DatasetInfo {
  id: string
  name: string
  documents: number
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
  // A vocabulary built by a `tokenize` job, for the run to train against. Absent
  // (or empty) means the run learns one from its own datasets, which is what
  // every run did before builds existed — and `max_vocab` is ignored whenever one
  // is named, since a vocabulary's size belongs to the build.
  //
  // Pretrain only, and only when starting from scratch: a fine-tune loads the
  // base model's own tokenizer, and a resumed pretrain (`resume_from` set)
  // loads the checkpoint's — both for the reason weights meeting token ids they
  // were never trained on produce confident nonsense, and both the composer's
  // to leave alone rather than to state (see `RESUMED_TOKENIZER_FIELDS`).
  tokenizer_id?: string
  // Which data-engine datasets to train on, each with how many of its documents
  // to take (0 = all of that dataset). Pretrain only.
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
  // A model to carry on from — a finished model or any of its checkpoints, since
  // both are a directory of the same shape and the difference is only whether a
  // `train_state.pt` sits in it.
  //
  // Two behaviours from one field, and the run reports which it took: a matching
  // corpus and schedule restores the optimizer, the RNG and the data order and
  // fast-forwards to the saved step; anything else carries the weights and the
  // moments onto this run's own schedule. Architecture comes from the state, and
  // a request that restates it differently is refused rather than overridden —
  // which is why the composer leaves the architecture fields alone rather than
  // rewriting them to the resumed model's.
  //
  // Pretrain only. `SftConfig` names a `base_model` instead and simply does not
  // have this field, the same way it has no `tokenizer_id` or `curriculum`.
  resume_from?: string
}

// The architecture a resumed run takes from the checkpoint instead of from the
// request — the engine's `_ARCHITECTURE` (`llm/orchestrate/train.py`), less the
// two fields the composer has never offered (`rope_base`, `learned_temperature`).
//
// Refused rather than overridden, and pydantic tells a *stated* field from a
// defaulted one by `model_fields_set` — so one of these left at whatever the form
// happened to be showing is not a harmless extra in the payload, it is the run
// being rejected before it starts. Hence one list: the composer reads it to know
// what not to show, `lib/jobs.ts` reads it to know what not to send.
export const RESUMED_ARCH_FIELDS = [
  "dimensions",
  "ctx_window",
  "heads",
  "layers",
  "ff_multiplier",
] as const satisfies readonly (keyof TrainConfig)[]

// The vocabulary a resumed run inherits, the same way `RESUMED_ARCH_FIELDS`
// covers the shape — the engine's `resumed_tokenizer` refuses a *stated*
// `tokenizer_id` that disagrees with the checkpoint's rather than honouring
// it, so the composer must neither show a control for this nor send it. Held
// apart from `RESUMED_ARCH_FIELDS` because neither is an architecture field —
// the composer renders `tokenizer_id` as its own Tokenizer selector, not
// through `ARCH_FIELDS` — but the two lists are read together wherever a
// resume's payload is stripped.
//
// `max_vocab` rides along because it is the *other* half of the same
// inheritance and the worse-behaved half: the engine's `tokenizer_for` returns
// the checkpoint's vocabulary before `max_vocab` is ever read, so a stated
// ceiling is not refused the way a disagreeing `tokenizer_id` is — it is
// silently ignored, then stored on the run and recapped as though it applied.
export const RESUMED_TOKENIZER_FIELDS = [
  "tokenizer_id",
  "max_vocab",
] as const satisfies readonly (keyof TrainConfig)[]

// One checkpoint, announced as its weights land: the engine's `Checkpoint`
// frame (`llm/train/progress.py`), arriving on the job socket as `checkpoint`.
//
// A whole model, so `model_id` is a model id like any other — it is what makes a
// listed checkpoint something you can resume from, chat with or benchmark.
export interface CheckpointFrame {
  model_id: string
  /** 1-based ordinal within the run — the numbered suffix a person reads. */
  index: number
  step: number
  total_steps: number
  epoch: number
  stage: number
  stage_name: string | null
  /** The training state file, which is what a later run resumes from. */
  path: string
  loss: number | null
  /** The run's own finished model rather than an intermediate. */
  final: boolean
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

// One source's share of the packed corpus. Fertility is per source because it is
// uneven across sources — code encodes at ~2 chars a token where prose runs at ~4
// — and that unevenness is how the context budget is really being spent.
export interface Fertility {
  dataset: string
  chars_per_token: number
  tokens: number
}

// One learned merge. `count` is how often the merged token occurs in the packed
// corpus, NOT the pair frequency that earned the merge — the tokenizers library
// keeps that only while it is choosing. See the engine's `bpe.merge_sequence`.
export interface Merge {
  token: string
  count?: number
}

// One document's presence inside one sampled training window. `start`/`length`
// are tokens from the window's first slot, clipped to it — so the window's own
// width is what places them, and that width is the run's `ctx_window` rather than
// anything on the frame (one packed corpus is legitimately trained at two).
//
// `opens`/`closes` are what the clipping loses: whether the document's own first
// and last token are in this window. An edge that is neither is the window's edge
// and not a boundary the model attends across — drawing the two alike is what
// turns one book into nine unrelated documents.
export interface RibbonSegment {
  dataset: number
  start: number
  length: number
  opens: boolean
  closes: boolean
}

// `overflow` is the documents past the engine's per-window cap. Non-zero means
// `segments` stops short of the window's width on purpose: the tail is real and
// too fine to draw as bars.
export interface RibbonWindow {
  index: number
  segments: RibbonSegment[]
  overflow: number
}

// The concrete picture behind `boundary_straddle`. `datasets` is the name table
// `RibbonSegment.dataset` indexes. Bounded by two engine-side constants
// (`RIBBON_WINDOWS` × `RIBBON_SEGMENTS`), never by the size of the corpus.
export interface Ribbon {
  datasets: string[]
  windows: RibbonWindow[]
}

// One source as a build is working through it: a dataset within a curriculum
// stage. `expected` is what the selection says it holds, `documents` how many of
// them have been read — the same dataset appears once per stage that names it,
// because a stage that trains on it again reads it again.
//
// `chars`/`tokens` are the live halves of `Fertility`: their ratio is this
// source's chars per token, available while the pack is running rather than only
// in the sidecar it writes at the end.
export interface BuildSource {
  dataset: string
  stage: number
  expected?: number
  documents: number
  chars: number
  tokens: number
}

// Which of phase 1's steps a build is on. Named by the engine
// (`llm.train.progress.BUILD_SPANS`) rather than inferred here from which counts
// have arrived: packing is the longest step of a run and used to report nothing
// at all, so "the step before the first one that has not reported" read the whole
// of it as a tokenizer that had already finished.
export type BuildStep =
  | "sampling"
  | "merges"
  | "packing"
  | "rendering"
  | "reusing"
  | "measuring"

export interface TokenizeStats {
  // Which step is running, and the clock it has been running on. `elapsed_secs`
  // is phase 1's own — a run's `TrainStats.elapsed_secs` starts at the first
  // training step, which is the moment all of this is over.
  step: BuildStep
  elapsed_secs: number
  // What is left of the step in flight, projected from the rate it has managed
  // so far. Absent until something has been read: a rate of nothing is not slow,
  // it is unmeasured.
  eta_secs: number
  // Documents encoded so far, against `documents`, which a build reports as the
  // count it set out to read before it reads any of them. On a fine-tune these
  // are conversation rows read, and `conversations` counts the ones that
  // survived rendering — the gap between the two is what was skipped.
  documents_done: number
  // Tokens a second over the last few seconds rather than over the build. The
  // average is what `eta_secs` is projected from and is deliberately smooth;
  // this is the one that says whether the thing is still moving right now.
  tokens_per_sec: number
  // Text drawn for the vocabulary's merges, against the budget it is drawn to.
  // Only the sampling step measures these — learning the merges from them
  // reports neither, because the library exposes no callback to observe it.
  sampled_bytes: number
  sample_budget: number
  // Per-source progress while a corpus is packed, in reading order. Bounded by
  // the engine, which sends a window centred on the source in flight;
  // `sources_total` is how many there really are.
  sources: BuildSource[]
  sources_total: number
  // Pretrain phase-1 stats (BPE tokenizer + corpus build).
  //
  // `documents` rather than `files`: the text arrives from the data engine a page
  // at a time and there are no files to count. `files` is kept for a run queued
  // before that, whose stored frames still have to render.
  documents: number
  files: number
  chars: number
  vocab: number
  merges: number
  // What the build produced, as opposed to how much of it there was. Every one is
  // measured off the packed corpus and its document table, so a run that reused a
  // build reports them exactly as the run that made it did.
  //
  // Most tokens first, and bounded by the engine.
  fertility: Fertility[]
  // Token ids the corpus never uses. The embedding table is tied, so each one is a
  // row read in every forward pass and trained by nothing — read against `vocab`.
  dead_vocab: number
  // The head of the merge list, in the order BPE learned it.
  merge_sequence: Merge[]
  // The window the three fields below were measured against, carried with them.
  // Not read from the run's config: a training job's run row has no config for
  // the panel to read, so that source is always absent and every window-relative
  // figure silently sized itself against zero.
  ctx_window: number
  // The fraction of `ctx_window` training windows holding an <|endoftext|>: how
  // often the model attends across a document boundary. A packing measure, and
  // the only direct one there is.
  boundary_straddle: number
  // Document lengths against `ctx_window` — fixed-width bins reaching two windows,
  // as fractions summing to ~1, the last bin absorbing everything beyond — plus
  // the share of documents no single window can hold. The pretrain counterpart of
  // `skipped_too_long`, except nothing is skipped: the document is split across
  // windows instead.
  doc_length_hist: number[]
  docs_over_ctx: number
  // A sample of actual training windows, each broken into the documents inside
  // it — what `boundary_straddle` looks like rather than what it measures.
  span_ribbon: Ribbon
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

  // ── Fine-tuning ───────────────────────────────────────────────────────────
  // Every field below is OPTIONAL because the engine omits what it did not
  // measure rather than sending a zero (see `Diagnostics.as_dict`): a pretrain
  // run carries none of them, and an SFT run carries only the ones it could
  // reach. `undefined` means "not measured" and must render as an absent figure,
  // never as a point at 0 — a zero here reads as a perfect loss.

  // Cross-entropy on a fixed batch of the base model's pretraining text, which
  // an SFT run never trains on. Climbing is catastrophic forgetting; this is the
  // number that says when to stop.
  pretrain_loss?: number
  // The diagnosed window's loss split by the SFT mask: the assistant tokens the
  // run takes gradient on, and the prompt region it only conditions on.
  loss_masked?: number
  loss_unmasked?: number
  // Mean trainable-token loss per equal slice of the context window, front to
  // back. A fixed 16 buckets whatever `ctx_window` the run trains at; a bucket
  // with no trainable token in the window is null.
  loss_by_position?: (number | null)[]
  // The hardest conversations in the diagnosed window, loss-descending, paired
  // index-for-index with the corpus offset that names each. The same
  // conversation is the same offset for the whole run.
  conversation_loss?: number[]
  conversation_start?: number[]
  // Mean KL(base ‖ tuned) in nats per token on a fixed held-out chat batch —
  // how far the tune has moved from the model it started as. Present only when
  // the run opted in: it keeps a second copy of the base model resident.
  base_kl?: number

  // ── Reinforcement learning ────────────────────────────────────────────────
  // Present only on an `rl` run and absent — not zero — on every other, the same
  // rule the fine-tuning fields above follow. A batch is a set of groups; a
  // group is G rollouts of one task, and its advantage is how much better a
  // rollout did than its siblings.

  // Fraction of episodes that answered correctly. The number an RL run is read
  // by, in place of a loss.
  pass_rate?: number
  mastery?: import("@/lib/practice").SkillProgress[]
  curriculum_events?: import("@/lib/practice").MasteryEvent[]
  test_before?: import("@/lib/practice").PracticeTest
  test_after?: import("@/lib/practice").PracticeTest
  practice_phase?: string
  // Fraction of groups that had ANY variance in their returns. A group whose
  // rollouts all passed or all failed has an exactly zero gradient, so this —
  // not the pass rate — is what says whether a run is learning or merely
  // spending. A run whose signal is collapsing is stalled however good its
  // pass rate looks.
  signal?: number
  groups?: number
  skipped_groups?: number
  // Mean episode return: the score plus its clipped shaping.
  mean_return?: number
  // KL(policy ‖ the model this run started from), in nats per drawn token. The
  // anchor, not a diagnostic: without it a binary reward and a forgiving grader
  // converge on a model that has forgotten how to write a sentence.
  kl?: number
  episodes?: number
  // Episodes the driver gave up on — a wedged sandbox, not a model that failed.
  // They are absent from the pass rate rather than counted as zeros.
  abandoned?: number
  episode_rounds?: number
  policy_tokens?: number
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
  // Wall-clock since the run's first step. The counterpart to `eta_secs`, and
  // what a finished run has instead of one.
  elapsed_secs: number
  epoch_done: boolean
  epoch_duration_secs: number
  // The state file written on this step, when one was. Not a run's list of
  // checkpoints: a save deduplicated against another rule leaves its frame with
  // nothing here, so `CheckpointFrame` is what a listing reads.
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
