import { cn } from "@/lib/utils"
import type { ModelTier } from "@/hooks/use-models"

// Accent color per tier, pulled from the theme's chart tokens so it tracks
// the kanagawa palette in both light (lotus) and dark (dragon) modes.
const ACCENTS: Record<ModelTier, string> = {
  flagship: "text-chart-1", // gold
  balanced: "text-chart-3", // blue
  fast: "text-chart-2", // green
}

export function ModelGlyph({
  tier,
  className,
}: {
  tier: ModelTier
  className?: string
}) {
  return (
    <div
      className={cn(
        "relative h-28 w-full overflow-hidden bg-muted/40",
        ACCENTS[tier],
        className
      )}
    >
      <svg
        viewBox="0 0 400 140"
        preserveAspectRatio="xMidYMid slice"
        className="h-full w-full"
        fill="none"
        aria-hidden
      >
        {tier === "flagship" && <Sun />}
        {tier === "balanced" && <Wave />}
        {tier === "fast" && <Streak />}
      </svg>
    </div>
  )
}

// Rising-sun: concentric arcs radiating from the bottom edge.
function Sun() {
  return (
    <g stroke="currentColor" strokeWidth="1.5">
      {[26, 52, 78, 104, 130].map((r, i) => (
        <circle
          key={r}
          cx="200"
          cy="140"
          r={r}
          strokeOpacity={0.45 - i * 0.07}
        />
      ))}
      <circle cx="200" cy="140" r="12" fill="currentColor" stroke="none" />
    </g>
  )
}

// Stacked waves — a nod to kanagawa's "Great Wave".
function Wave() {
  return (
    <g stroke="currentColor" strokeWidth="1.5" fill="none">
      {[0, 1, 2, 3].map((i) => (
        <path
          key={i}
          transform={`translate(0 ${24 + i * 26})`}
          d="M-20 30 C 40 0, 80 60, 140 30 S 240 0, 300 30 S 400 60, 460 30"
          strokeOpacity={0.5 - i * 0.09}
        />
      ))}
    </g>
  )
}

// Speed streaks — diagonal beams suggesting low latency.
function Streak() {
  return (
    <g stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      {Array.from({ length: 8 }).map((_, i) => (
        <line
          key={i}
          x1={i * 56 - 60}
          y1="150"
          x2={i * 56 + 50}
          y2="-10"
          strokeOpacity={i % 2 === 0 ? 0.4 : 0.22}
        />
      ))}
    </g>
  )
}
