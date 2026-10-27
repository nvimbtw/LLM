import type { Model } from "@/hooks/use-models"

/** Resolves the actual request/meter multiplier from public implemented capability metadata; unavailable/custom support retains the trained window. */
export function effectiveContextMultiplier(
  model: Model | null | undefined,
  requested: number
): number {
  return model?.supportsContextExtension === true &&
    Number.isFinite(requested) &&
    requested >= 1
    ? requested
    : 1
}
