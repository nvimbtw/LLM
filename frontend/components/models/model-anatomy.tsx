import type { ModelDetail } from "@/hooks/use-model"
import { formatCount, formatNumber, formatParams } from "./format"

/** Places resolved legacy shape facts on their operations; modular and missing shapes remain explicit. */
export function ModelAnatomy({ model }: { model: ModelDetail }) {
  const shape = model.definitionId ? null : model.architecture
  const vocabulary =
    model.vocabularySize === null
      ? "Unavailable vocabulary"
      : `${formatNumber(model.vocabularySize)}-token vocabulary`
  return (
    <section aria-labelledby="model-anatomy-title">
      <div className="mb-6 flex flex-wrap items-baseline justify-between gap-4">
        <h2 id="model-anatomy-title" className="font-display text-2xl">
          Architecture
        </h2>
        <p className="text-xs text-muted-foreground">
          <strong className="mr-2 font-display text-2xl font-normal text-foreground">
            {formatParams(model.params, model.parameterCountEstimated)}
          </strong>
          parameters{model.parameterCountEstimated ? " · estimated" : ""}
        </p>
      </div>
      <p className="mb-5 text-center text-xs text-muted-foreground">
        {model.contextWindow > 0
          ? `${formatNumber(model.contextWindow)}-token context`
          : "Context unavailable"}
      </p>
      {shape ? (
        <div
          className="grid items-center gap-5 sm:grid-cols-[1fr_auto_1.5fr_auto_1fr]"
          aria-label="Simplified Transformer structure"
        >
          <div className="text-center">
            <h3 className="text-sm">Embedding</h3>
            <div
              aria-hidden="true"
              className="mx-auto my-4 grid w-fit grid-cols-7 gap-1"
            >
              {Array.from({ length: 28 }, (_, i) => (
                <i
                  key={i}
                  className="size-1.5 bg-tier-med"
                  style={{ opacity: 0.2 + ((i * 13) % 9) / 14 }}
                />
              ))}
            </div>
            <p className="text-xs text-muted-foreground">{vocabulary}</p>
            <p className="mt-1 text-xs text-muted-foreground">
              {formatNumber(shape.dimensions)} channels
            </p>
          </div>
          <AnatomyArrow />
          <div className="rounded-sm border border-border px-5 py-4">
            <div className="mb-5 flex flex-wrap items-baseline justify-between gap-2 text-sm">
              <h3>Transformer</h3>
              <span className="text-xs text-tier-med">
                {formatNumber(shape.layers)} layers
              </span>
            </div>
            <div className="text-center">
              <h4 className="text-sm">Causal attention</h4>
              <div
                aria-hidden="true"
                className="my-3 flex justify-center gap-1"
              >
                {Array.from({ length: Math.min(shape.heads, 12) }, (_, i) => (
                  <i key={i} className="h-4 w-1 bg-primary/50" />
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                {formatNumber(shape.heads)} heads
                {shape.dimensions % shape.heads === 0
                  ? ` · ${formatNumber(shape.dimensions / shape.heads)} channels / head`
                  : " · head width unavailable"}
              </p>
              <span
                aria-hidden="true"
                className="my-3 block text-muted-foreground"
              >
                ↓
              </span>
              <h4 className="text-sm">Feed-forward</h4>
              <div
                aria-hidden="true"
                className="my-3 flex h-6 items-end justify-center gap-1"
              >
                {[8, 13, 19, 24, 19, 13, 8].map((height, i) => (
                  <i
                    key={i}
                    className="w-1 bg-tier-med/60"
                    style={{ height }}
                  />
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                {formatNumber(shape.dimensions * shape.ff_multiplier)} hidden ·{" "}
                {shape.ff_multiplier}×
              </p>
            </div>
          </div>
          <AnatomyArrow />
          <div className="text-center">
            <h3 className="text-sm">Token scores</h3>
            <div
              aria-hidden="true"
              className="my-4 flex h-12 items-end justify-center gap-1"
            >
              {[7, 15, 29, 43, 22, 12, 5].map((height, i) => (
                <i key={i} className="w-1.5 bg-primary/50" style={{ height }} />
              ))}
            </div>
            <p className="text-xs text-muted-foreground">
              {model.vocabularySize === null
                ? "Output count unavailable"
                : `${formatNumber(model.vocabularySize)} outputs`}
            </p>
          </div>
        </div>
      ) : (
        <div className="space-y-3 py-6 text-sm text-muted-foreground">
          <p>
            {model.definitionId
              ? "This model uses a saved modular architecture."
              : "Architecture metadata is unavailable."}
          </p>
          <p className="text-xs">{vocabulary}</p>
        </div>
      )}
    </section>
  )
}

/** Connects the simplified operations in horizontal and narrow vertical layouts. */
function AnatomyArrow() {
  return (
    <span
      aria-hidden="true"
      className="rotate-90 text-center text-xl text-muted-foreground sm:rotate-0"
    >
      →
    </span>
  )
}

/** Summarizes only recorded run measurements, leaving missing or private provenance unavailable. */
export function TrainingRecord({ model }: { model: ModelDetail }) {
  const result = model.dataset
  const data = result?.dataset
  const figures = [
    ["epochs", model.config?.epochs],
    ["documents", data?.documents ?? result?.files],
    ["conversations", data?.conversations],
    ["characters", data?.chars ?? result?.chars],
    ["corpus tokens", data?.corpus_len ?? result?.corpus_len],
    ["BPE merges", result?.merges],
  ] as const
  const facts = figures.flatMap(([label, value]) =>
    typeof value === "number" && Number.isFinite(value) && value >= 0
      ? [`${formatCount(value)} ${label}`]
      : []
  )
  if (
    model.config?.batch_size !== undefined &&
    Number.isFinite(model.config.batch_size)
  )
    facts.push(`batch size ${formatCount(model.config.batch_size)}`)
  if (model.config?.lr !== undefined && Number.isFinite(model.config.lr))
    facts.push(`learning rate ${model.config.lr}`)
  return (
    <section className="max-w-2xl" aria-labelledby="training-record-title">
      <h2 id="training-record-title" className="mb-3 font-display text-2xl">
        Training record
      </h2>
      <p className="text-sm leading-7 text-muted-foreground">
        {facts.length
          ? facts.join(" · ")
          : "Training provenance is unavailable for this view."}
      </p>
    </section>
  )
}
