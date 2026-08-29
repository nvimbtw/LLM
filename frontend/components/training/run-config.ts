// What a training run's fields are, and what makes one valid.
//
// No JSX and no layout: the composer renders these as a full-width board and a
// follow-up step renders them as a narrow rail, and neither arrangement should
// be able to disagree with the other about what a learning rate may be.

import { DEFAULT_CONFIG } from "@/hooks/use-training"
import { isFineTune, type TrainConfig, type TrainMode } from "@/lib/training"

// Numeric fields split by which mode they apply to. Architecture + dataset knobs
// only make sense for a fresh pretrain; a fine-tune inherits the base model's
// architecture, so in SFT mode only the run knobs (epochs, learning rate) show.
// `pow2` fields only accept powers of two, so scrolling / arrowing them cycles
// through 16 · 32 · 64 · 128 … rather than stepping by one (see Pow2Input).
// `maxExp` overrides the default exponent ceiling for fields that legitimately go
// larger (e.g. vocab).
export type Field = {
  key: keyof TrainConfig
  label: string
  /** What the number means, for the one-line note under it on the board. */
  hint?: string
  step?: number
  pow2?: boolean
  maxExp?: number
}

// Every field the rail edits as text. Everything else goes through `Number()`,
// so a string field missing from here arrives at the engine as 0 — silently, and
// only for the one field somebody forgot.
export const STRING_FIELDS = new Set<keyof TrainConfig>([
  "model_name",
  "base_model_id",
  "tokenizer_id",
  "resume_from",
])

export const ARCH_FIELDS: Field[] = [
  {
    key: "dimensions",
    label: "Dimensions",
    hint: "Width of the residual stream",
    pow2: true,
  },
  {
    key: "ctx_window",
    label: "Context window",
    hint: "Tokens the model sees at once",
    pow2: true,
  },
  { key: "heads", label: "Heads", hint: "Attention heads per layer", pow2: true },
  { key: "layers", label: "Layers", hint: "Transformer blocks, stacked" },
  {
    key: "ff_multiplier",
    label: "FF multiplier",
    hint: "Feed-forward width, as a multiple",
  },
  // Vocab can be far larger than a model dimension — GPT-2 is ~50k (2^16). Allow
  // up to 2^20 (~1M).
  {
    key: "max_vocab",
    label: "Max vocab",
    hint: "Ceiling on learned merges",
    pow2: true,
    maxExp: 20,
  },
]

export const RUN_FIELDS: Field[] = [
  { key: "epochs", label: "Epochs", hint: "Passes over the corpus" },
  { key: "lr", label: "Learning rate", hint: "Step size at each update", step: 0.001 },
]

// SFT wants a much lower learning rate than a from-scratch run — the base model
// is already near a good minimum, so a large step would wash out what it learned.
// Now that the pretrain default is 1e-4 (ISSUE-0052/0058 lifted the LR cap), SFT
// sits an order of magnitude below it at 1e-5.
export const SFT_DEFAULT_LR = 0.00001

// KTO sits lower still. Its gradient is the cross-entropy one scaled by a factor
// that can exceed 1 on a confidently-wrong example, and it is deliberately allowed
// to push probability *down* — an over-large step there degrades the model far
// faster than an over-large SFT step, which can only ever pull it toward real text.
export const KTO_DEFAULT_LR = 0.000005

// RAFT is ordinary SFT under the hood — same loss, same masking — so it wants the
// SFT learning rate, not KTO's much lower one. Named separately because reading
// `SFT_DEFAULT_LR` under a RAFT branch invites someone to "fix" it to the KTO
// value. These two are only reached now when the form is seeded from a cycle's
// job; the cycle's own runs use `raft_lr` / `kto_lr` from its config instead.
export const RAFT_DEFAULT_LR = SFT_DEFAULT_LR

// The KL anchor's default. See llm::kto::KtoParams::beta.
export const KTO_DEFAULT_BETA = 0.1

// Every learning rate the mode toggle sets on the user's behalf, so switching modes
// can tell "still on a default" from "the user chose this".
export const MODE_DEFAULT_LRS: number[] = [
  DEFAULT_CONFIG.lr,
  SFT_DEFAULT_LR,
  KTO_DEFAULT_LR,
]

export function defaultLrFor(mode: TrainMode): number {
  if (mode === "sft") return SFT_DEFAULT_LR
  if (mode === "raft") return RAFT_DEFAULT_LR
  if (mode === "kto") return KTO_DEFAULT_LR
  return DEFAULT_CONFIG.lr
}

export function isPowerOfTwo(n: number) {
  return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0
}

// "No tokenizer chosen" needs a value of its own: an empty string is how a
// Select says *nothing is selected*, and this option is a real choice — the one
// every run made before vocabulary builds existed. Mapped back to `undefined`
// on the way out, which is what the config means by it.
export const LEARN_WITH_RUN = "learn-with-run"

// "Start from random weights", for the same reason `LEARN_WITH_RUN` exists: it
// is a real choice — the one every run made before checkpoints could be picked
// up — and the empty string is how a Select says *nothing is selected*. Mapped
// back to `undefined` on the way out, which is what the config means by it.
export const FROM_SCRATCH = "from-scratch"

// "The finished weights", as opposed to one of the model's intermediate saves.
// Selecting it means `resume_from` is the parent model's own id.
export const FINISHED_WEIGHTS = "finished-weights"

// Name is a display label only (weights live under the model's UUID), so spaces
// and dots like "Opal 1.1" are allowed — just no surrounding whitespace or
// control characters. Shared by both modes.
function validateName(name: string): string[] {
  if (!name.trim()) return ["Model name is required."]
  if (name !== name.trim())
    return ["Model name may not have leading or trailing whitespace."]
  if (/[\x00-\x1f\x7f]/.test(name))
    return ["Model name may not contain control characters."]
  return []
}

// Mirrors the backend's validate_config / validate_sft so the user sees the
// constraints before submitting. Returns a list of human-readable problems.
export function validate(cfg: TrainConfig, inheritsBase = false): string[] {
  const errs = validateName(cfg.model_name)

  if (isFineTune(cfg.mode)) {
    // A follow-up has no base to select: it fine-tunes whatever the job before it
    // produces, which is exactly why it is a follow-up.
    if (!cfg.base_model_id && !inheritsBase)
      errs.push("Select a base model to fine-tune.")
    if (
      cfg.mode === "kto" &&
      !(Number.isFinite(cfg.beta) && (cfg.beta ?? 0) > 0)
    ) {
      errs.push("Beta must be a positive number.")
    }
    if (!Number.isFinite(cfg.epochs) || cfg.epochs <= 0)
      errs.push("Epochs must be positive.")
    if (!Number.isFinite(cfg.lr) || cfg.lr <= 0)
      errs.push("Learning rate must be positive.")
    // A plain fine-tune trains on conversations it names. RAFT and KTO train on
    // the model's own generations instead, so a dataset selection there is not a
    // thing to require. The engine refuses a run that names nothing — this is
    // what stops that arriving as a job that queues and then fails.
    if (cfg.mode === "sft" && (cfg.datasets ?? []).length === 0)
      errs.push("Select at least one dataset to fine-tune on.")
    return errs
  }

  const positives: [keyof TrainConfig, string][] = [
    ["dimensions", "Dimensions"],
    ["ctx_window", "Context window"],
    ["heads", "Heads"],
    ["layers", "Layers"],
    ["epochs", "Epochs"],
    ["ff_multiplier", "FF multiplier"],
  ]
  for (const [k, label] of positives) {
    const v = cfg[k] as number
    if (!Number.isFinite(v) || v <= 0) errs.push(`${label} must be positive.`)
  }

  if (!isPowerOfTwo(cfg.dimensions))
    errs.push("Dimensions must be a power of two (16, 32, 64, 128…).")
  if (!isPowerOfTwo(cfg.ctx_window))
    errs.push("Context window must be a power of two (16, 32, 64, 128…).")
  if (
    Number.isFinite(cfg.dimensions) &&
    cfg.heads > 0 &&
    cfg.dimensions % cfg.heads !== 0
  )
    errs.push(
      `Heads must divide dimensions evenly. ${cfg.dimensions} ÷ ${cfg.heads} isn't whole.`
    )
  if (!Number.isFinite(cfg.lr) || cfg.lr <= 0)
    errs.push("Learning rate must be positive.")
  if (!Number.isFinite(cfg.max_vocab) || cfg.max_vocab < 256)
    errs.push("Max vocab must be at least 256.")

  // Curriculum runs supply their datasets per stage; a plain run uses the flat
  // selection. Validate whichever is active (mirrors the backend).
  const curriculum = cfg.curriculum ?? []
  if (curriculum.length > 0) {
    curriculum.forEach((stage, i) => {
      const ds = stage.datasets ?? []
      if (!Number.isFinite(stage.epochs) || stage.epochs <= 0)
        errs.push(`Stage ${i + 1}: epochs must be positive.`)
      if (ds.length === 0)
        errs.push(`Stage ${i + 1}: select at least one dataset.`)
      if (ds.some((d) => !Number.isInteger(d.count) || d.count < 0))
        errs.push(
          `Stage ${i + 1}: each dataset count must be 0 (all) or positive.`
        )
    })
  } else {
    // Defensive: a config from a stale source (e.g. a preset saved before this
    // field existed) may lack `datasets`; treat missing as an empty selection so
    // it surfaces as a normal validation error rather than a crash.
    const selected = cfg.datasets ?? []
    if (selected.length === 0)
      errs.push("Select at least one dataset to train on.")
    if (selected.some((d) => !Number.isInteger(d.count) || d.count < 0))
      errs.push(
        "Each dataset count must be 0 (all) or a positive whole number."
      )
  }
  return errs
}

/** The numeric fields a mode actually shows, in the order they are rendered. */
export function fieldsFor(cfg: TrainConfig): { arch: Field[]; run: Field[] } {
  const curriculumOn = (cfg.curriculum?.length ?? 0) > 0
  // In curriculum mode, epochs are per-stage, so drop the run-wide Epochs field.
  const run = curriculumOn
    ? RUN_FIELDS.filter((f) => f.key !== "epochs")
    : RUN_FIELDS

  if (cfg.mode === "kto") {
    return {
      arch: [],
      run: [...RUN_FIELDS, { key: "beta", label: "Beta (KL anchor)", hint: "How hard the KL anchor pulls", step: 0.05 }],
    }
  }
  if (isFineTune(cfg.mode)) return { arch: [], run: RUN_FIELDS }

  // A named vocabulary carries its own size, so the field would be a control
  // wired to nothing — the engine ignores `max_vocab` whenever one is named.
  return {
    arch: ARCH_FIELDS.filter(
      (f) => !(f.key === "max_vocab" && cfg.tokenizer_id)
    ),
    run,
  }
}

/** Head size and feed-forward width, or null where the config cannot say. */
export function derived(cfg: TrainConfig): {
  headDim: number | null
  dFf: number | null
} {
  return {
    headDim:
      cfg.heads > 0 && cfg.dimensions % cfg.heads === 0
        ? cfg.dimensions / cfg.heads
        : null,
    dFf: cfg.dimensions * cfg.ff_multiplier || null,
  }
}
