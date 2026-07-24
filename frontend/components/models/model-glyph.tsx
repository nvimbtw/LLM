import { cn } from "@/lib/utils"
import type { ModelTier } from "@/hooks/use-models"

// Tier → how many of the three steps are lit. The ascending heights read as a
// size scale; the lit count is this model's place on it.
const LEVEL: Record<ModelTier, number> = {
  fast: 1,
  balanced: 2,
  flagship: 3,
}

// A compact three-step size meter — the editorial replacement for the old motif
// banners. Lit steps use the single accent; the rest are hairline ink.
export function ModelSizeMeter({
  tier,
  className,
}: {
  tier: ModelTier
  className?: string
}) {
  const level = LEVEL[tier]
  return (
    <div className={cn("flex items-end gap-[3px]", className)} aria-hidden>
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className={cn(
            "w-[3px] rounded-full",
            i < level ? "bg-primary" : "bg-border"
          )}
          style={{ height: `${8 + i * 5}px` }}
        />
      ))}
    </div>
  )
}
