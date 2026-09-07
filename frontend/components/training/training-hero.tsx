"use client"

import { Progress } from "@/components/ui/progress"
import { Throbber, type ThrobberMode } from "@/components/ui/throbber"
import { duration, StatRun, type Stat } from "@/components/training/stat-run"
import { cn } from "@/lib/utils"
import type {
  CurriculumStage,
  RunPhase,
  TokenizeStats,
  TrainStats,
} from "@/lib/training"

function compact(n: number | undefined | null): string {
  if (n == null) return "—"
  return Intl.NumberFormat("en-US", { notation: "compact" }).format(n)
}

function loss4(v: number | undefined): string {
  return v != null && Number.isFinite(v) ? v.toFixed(4) : "—"
}

/**
 * Which mark the hero corner shows for a run's phase.
 *
 * Deliberately not one throbber mode per `RunPhase`: the mark says what the GPU is
 * doing, and phase 1 (building a dataset) is not training however the run labels
 * it. A finished run keeps the mark rather than dropping it, at rest — an empty
 * corner where something was reads as a component that failed to render.
 */
function phaseMode(status: RunPhase | null): ThrobberMode {
  switch (status) {
    case "tokenizing":
      return "tokenizing"
    case "training":
      return "training"
    default:
      return "idle"
  }
}

// The headline and the line under it: what the run is doing, said once, in
// words. Every branch returns both, because a heading with nothing beneath it
// leaves a hole the eye reads as missing data rather than as an absence.
function describe(
  status: RunPhase | null,
  isSft: boolean,
  tokenize: Partial<TokenizeStats>,
  train: Partial<TrainStats>
): { headline: string; detail: string } {
  switch (status) {
    case "tokenizing":
      return {
        headline: isSft ? "Building the dataset" : "Building the corpus",
        detail: buildDetail(isSft, tokenize),
      }
    case "training": {
      const stage = train.stage_name?.trim()
      const epoch = `Epoch ${train.epoch ?? 0} of ${train.total_epochs ?? 0}`
      return {
        headline: isSft ? "Fine-tuning" : "Training",
        detail: stage ? `${epoch} · ${stage}` : epoch,
      }
    }
    case "done":
      return {
        headline: "Trained",
        detail: `Finished at loss ${loss4(train.loss)} after ${train.total_epochs ?? 0} epoch(s)`,
      }
    case "cancelled":
      return {
        headline: "Cancelled",
        detail:
          "Stopped at a step boundary — the weights it had are consistent.",
      }
    case "failed":
      return {
        headline: "Failed",
        detail: "The header above carries what went wrong.",
      }
    default:
      return { headline: "Waiting", detail: "This run has not started yet." }
  }
}

// What the build is doing, in a sentence, off the step the engine names rather
// than off which counters happen to have arrived. The line it replaces was
// written once — when the tokenizer finished — and then stood unchanged for the
// hours the pack took, which is the whole of what made a working build
// indistinguishable from a stuck one.
function buildDetail(isSft: boolean, tokenize: Partial<TokenizeStats>): string {
  const read = tokenize.documents_done
  const expected = tokenize.documents ?? tokenize.files
  // "N of M" once both are known, "M" while only the target is, and nothing at
  // all before either — a count of "—" in the middle of a sentence reads as a
  // bug rather than as a figure that has not arrived.
  const through =
    expected == null
      ? null
      : read != null
        ? `${compact(read)} of ${compact(expected)}`
        : compact(expected)

  switch (tokenize.step) {
    case "sampling":
      return through
        ? `Drawing the vocabulary's sample from ${through} document(s)`
        : "Drawing the text the vocabulary's merges are learned from…"
    case "merges":
      return "Learning the merges — one call into the tokenizer, which reports nothing until it returns"
    case "packing":
      return tokenize.total_tokens != null
        ? `Encoding ${through ?? "the corpus"} — ${compact(tokenize.total_tokens)} token(s) so far`
        : `Encoding ${through ?? "the corpus"} into the packed corpus`
    case "rendering":
      return through
        ? `Rendering ${through} conversation(s) into a masked dataset`
        : "Rendering chat conversations into a masked dataset…"
    case "reusing":
      return isSft
        ? "Reusing the dataset an earlier run rendered"
        : "Reusing the corpus an earlier run packed"
    case "measuring":
      return `Measuring what was built — ${compact(tokenize.documents)} document(s), ${compact(tokenize.total_tokens)} token(s)`
    default:
      // A build whose first frame has not landed, or one queued before the
      // engine named its steps: say what it is about to do rather than nothing.
      return isSft
        ? "Rendering chat conversations into a masked dataset…"
        : "Collecting documents…"
  }
}

// The curriculum figure: which stage the run is on plus a segmented bar — one
// segment per stage, sized by its epoch weight, filled by that stage's own
// progress. Completed stages read full, the current one fills live, upcoming ones
// sit empty; it doubles as a preview during phase 1 (all empty) and reads "all
// full" once the run is done. Progress comes from the global epoch position
// (monotonic across stages), so it stays smooth even though each stage reports
// its own chunk counts.
function CurriculumProgress({
  stages,
  train,
  done,
}: {
  stages: CurriculumStage[]
  train: Partial<TrainStats>
  done: boolean
}) {
  const weights = stages.map((s) => Math.max(1, s.epochs))
  const starts: number[] = []
  let acc = 0
  for (const w of weights) {
    starts.push(acc)
    acc += w
  }
  const totalEpochs = acc

  // Continuous position in epoch-units: completed global epochs plus the fraction
  // of the current epoch done (its chunk / the stage's chunk count). Pinned to the
  // end once the run finishes so every segment reads full.
  const chunkFrac =
    train.total_chunks && train.total_chunks > 0
      ? Math.min(1, (train.chunk ?? 0) / train.total_chunks)
      : 0
  const epochPos = done
    ? totalEpochs
    : Math.max(0, (train.epoch ?? 0) - 1 + chunkFrac)

  // The authoritative current stage comes from the engine's frame; fall back to
  // the epoch position (during phase 1, before the first training frame).
  const currentStage =
    train.stage != null
      ? train.stage
      : Math.min(
          stages.length - 1,
          starts.findIndex(
            (start, i) => epochPos >= start && epochPos < start + weights[i]
          )
        )
  const activeIdx = currentStage < 0 ? 0 : currentStage
  const activeName =
    stages[activeIdx]?.name?.trim() ||
    train.stage_name?.trim() ||
    `Stage ${activeIdx + 1}`

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-xs font-medium text-muted-foreground">
          Curriculum
        </span>
        <span className="min-w-0 truncate text-xs text-muted-foreground tabular-nums">
          Stage {activeIdx + 1} / {stages.length}
          <span className="mx-1.5 text-muted-foreground">·</span>
          <span className="text-foreground">{activeName}</span>
        </span>
      </div>

      <div className="flex items-stretch gap-1">
        {stages.map((stage, i) => {
          const frac = Math.min(
            1,
            Math.max(0, (epochPos - starts[i]) / weights[i])
          )
          const isActive = i === activeIdx && !done
          return (
            <div
              key={i}
              className="flex flex-col gap-1"
              style={{ flexGrow: weights[i], flexBasis: 0, minWidth: 0 }}
              title={`${stage.name?.trim() || `Stage ${i + 1}`} — ${stage.epochs} epoch${stage.epochs === 1 ? "" : "s"}`}
            >
              <div
                className={cn(
                  "h-2 overflow-hidden rounded-full bg-primary/15",
                  isActive && "ring-1 ring-primary/40"
                )}
              >
                <div
                  className="h-full rounded-full bg-primary transition-[width] duration-300"
                  style={{ width: `${frac * 100}%` }}
                />
              </div>
              <span
                className={cn(
                  "truncate text-xs tabular-nums",
                  isActive ? "text-foreground" : "text-muted-foreground"
                )}
              >
                {stage.name?.trim() || `Stage ${i + 1}`}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export interface TrainingHeroProps {
  status: RunPhase | null
  isSft: boolean
  tokenize: Partial<TokenizeStats>
  train: Partial<TrainStats>
  // The run's curriculum stages (empty for a plain single-pass run).
  curriculum: CurriculumStage[]
}

/**
 * The train page's hero: where the run says what it is doing and how far along
 * it is, in one block.
 *
 * A framed band spanning the full width rather than a column beside the config
 * card. The figures are the reason the page exists, and as a column they were
 * narrow enough that four of them wrapped — which is what made the top of this
 * page read as debris. Given the width they sit in one still row, the phase line
 * has room for a sentence, and the progress bar is long enough to be read as a
 * position rather than a decoration.
 *
 * Every region is always present and fills in: a run that has just started shows
 * the same layout as one an hour in, with "—" where a number has not arrived.
 * Nothing mounts or unmounts as the phases advance, so the page never jumps.
 */
export function TrainingHero({
  status,
  isSft,
  tokenize,
  train,
  curriculum,
}: TrainingHeroProps) {
  // The training figures are the headline from the moment training starts
  // through done; before that, phase 1's own counts are.
  const trainPhase = status != null && status !== "tokenizing"
  const finished =
    status === "done" || status === "cancelled" || status === "failed"
  const showCurriculum = status != null && (curriculum?.length ?? 0) > 1

  const { headline, detail } = describe(status, isSft, tokenize, train)

  const stats: Stat[] = trainPhase
    ? [
        {
          label: "Epoch",
          value: `${train.epoch ?? 0} / ${train.total_epochs ?? 0}`,
        },
        { label: "Loss", value: loss4(train.loss) },
        { label: "Avg loss", value: loss4(train.avg_loss) },
        // A run that has stopped has no estimate left, only a duration. The slot
        // stays where it is so the strip does not reshuffle when it lands.
        finished
          ? { label: "Elapsed", value: duration(train.elapsed_secs) }
          : { label: "ETA", value: duration(train.eta_secs) },
      ]
    : isSft
      ? [
          {
            label: "Conversations",
            // A preference tune names the same count `examples`; both are "how
            // many of the things this mode trains on did phase 1 find".
            value: compact(tokenize.conversations ?? tokenize.examples),
          },
          { label: "Total tokens", value: compact(tokenize.total_tokens) },
          { label: "Trainable", value: compact(tokenize.trainable_tokens) },
          { label: "Too long", value: compact(tokenize.skipped_too_long) },
        ]
      : [
          // Live counts rather than the four end-of-phase figures that used to
          // stand here reading "—" for the whole of the build. Vocabulary and
          // merges are in the build panel's step rail below, where they land the
          // moment the tokenizer is trained.
          {
            label: "Documents",
            value:
              tokenize.documents_done != null && tokenize.documents != null
                ? `${compact(tokenize.documents_done)} / ${compact(tokenize.documents)}`
                : compact(tokenize.documents ?? tokenize.files),
          },
          { label: "Tokens", value: compact(tokenize.total_tokens) },
          {
            label: "Tokens / sec",
            value: compact(tokenize.tokens_per_sec),
          },
          {
            label: "Time left",
            value: duration(tokenize.eta_secs),
          },
        ]

  const totalSteps =
    (train.total_epochs ?? 0) * (train.total_chunks ?? 0) || undefined
  const trainPct = totalSteps
    ? Math.min(100, Math.round(((train.step ?? 0) / totalSteps) * 100))
    : 0
  const pct = trainPhase ? trainPct : Math.round((tokenize.progress ?? 0) * 100)
  const position =
    trainPhase && totalSteps
      ? `Step ${(train.step ?? 0).toLocaleString()} of ${totalSteps.toLocaleString()}`
      : trainPhase
        ? "Waiting for the first report…"
        : tokenize.elapsed_secs != null
          ? `Phase 1 of 2 — ${duration(tokenize.elapsed_secs)} in`
          : "Phase 1 of 2 — preparing the data"

  return (
    <section className="rounded-(--radius-sheet) border border-border bg-card">
      <div className="flex items-start justify-between gap-6 p-5">
        <div className="min-w-0 space-y-1">
          <h2 className="font-heading text-xl font-semibold tracking-tight">
            {headline}
          </h2>
          <p className="text-sm text-balance text-muted-foreground">{detail}</p>
        </div>
        {/* The site's own mark, where a mini copy of the embedding globe used to
            sit. The globe is a *figure* — it shows the token cloud's principal
            axes — and up here, at 96px with no axes or caption, it read as
            decoration while the real one (with its explanation) sits in the
            diagnostics panel below. The throbber says the one thing this corner
            should: which phase the run is in, and that it is still moving. */}
        <Throbber
          mode={phaseMode(status)}
          className="hidden shrink-0 self-start sm:block"
        />
      </div>

      <div className="space-y-2 px-5 pb-5">
        <Progress value={pct} />
        <div className="flex items-baseline justify-between gap-3 text-xs text-muted-foreground">
          <span className="min-w-0 truncate tabular-nums">{position}</span>
          <span className="tabular-nums">{pct}%</span>
        </div>
      </div>

      {showCurriculum && (
        <div className="border-t border-border px-5 py-4">
          <CurriculumProgress
            stages={curriculum}
            train={train}
            done={status === "done"}
          />
        </div>
      )}

      <div className="border-t border-border">
        <StatRun stats={stats} />
      </div>
    </section>
  )
}
