import type { TrainConfig } from "./training"

export type ParameterShape = Pick<
  TrainConfig,
  "dimensions" | "layers" | "ff_multiplier"
>

/** Estimates the standard model's trainable parameters from its shape and vocabulary size. Counts tied embeddings once, four attention projections, both biased feed-forward projections and layer norms; RoPE adds no learned position table. Invalid or unsafe draft counts return null. */
export function estimateTrainingParameters(
  shape: ParameterShape,
  vocabularySize: number | null
): number | null {
  const { dimensions, layers, ff_multiplier } = shape
  if (vocabularySize === null) return null
  if (
    [dimensions, layers, ff_multiplier, vocabularySize].some(
      (value) => !Number.isSafeInteger(value) || value <= 0
    )
  )
    return null
  const hidden = dimensions * ff_multiplier
  const block =
    4 * dimensions ** 2 + 2 * dimensions * hidden + hidden + 5 * dimensions
  const parameters =
    dimensions * vocabularySize + layers * block + 2 * dimensions
  return Number.isSafeInteger(parameters) ? parameters : null
}
