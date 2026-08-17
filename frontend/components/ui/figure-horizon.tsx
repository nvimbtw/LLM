import { cn } from "@/lib/utils"

/**
 * A rule across the page, opened at the centre.
 *
 * The new-chat figure. Deliberately the quietest of the three: it sits directly
 * above a composer that has to stay the subject of the screen, so it is one line
 * and an aperture in it, and it fades out at both ends rather than stopping —
 * a hard stop would read as a divider, which is a different thing entirely.
 *
 * Static and pure, same terms as `figure-lattice.tsx` and `figure-seed.tsx`. The
 * fade is stepped segments rather than a `<linearGradient>` on purpose: a
 * gradient needs an `id`, and an `id` in a component is a collision waiting for
 * the second instance on a page.
 */

const WIDTH = 400
const HEIGHT = 120
const MID_X = WIDTH / 2
const MID_Y = 66
const GAP = 54
const SEGMENTS = 7
const RINGS = [15, 25, 36, 48]

// From the aperture outward to the edge, in equal steps. Opacity falls with
// distance, so the line arrives out of the paper instead of starting on it.
const SEGMENT_SPAN = (MID_X - GAP) / SEGMENTS

const RULE = Array.from({ length: SEGMENTS }, (_, i) => ({
  inner: GAP + i * SEGMENT_SPAN,
  outer: GAP + (i + 1) * SEGMENT_SPAN,
  opacity: 0.3 * (1 - i / SEGMENTS) ** 1.3,
}))

export interface FigureHorizonProps {
  className?: string
}

export function FigureHorizon({ className }: FigureHorizonProps) {
  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      fill="none"
      aria-hidden="true"
      focusable="false"
      className={cn("h-auto w-full text-foreground", className)}
    >
      <g stroke="currentColor" strokeWidth={1} strokeLinecap="round">
        {RULE.map(({ inner, outer, opacity }, i) => (
          <g key={i} strokeOpacity={opacity}>
            <line x1={MID_X - outer} y1={MID_Y} x2={MID_X - inner} y2={MID_Y} />
            <line x1={MID_X + inner} y1={MID_Y} x2={MID_X + outer} y2={MID_Y} />
          </g>
        ))}
      </g>

      {/* Half-rings only, sitting on the rule rather than crossing it, so the
          line still reads as continuous behind them. */}
      <g stroke="currentColor" strokeWidth={1} fill="none">
        {RINGS.map((r, i) => (
          <path
            key={r}
            d={`M ${MID_X - r} ${MID_Y} A ${r} ${r} 0 0 1 ${MID_X + r} ${MID_Y}`}
            strokeOpacity={0.26 * (1 - i / RINGS.length) ** 0.9}
          />
        ))}
      </g>

      <g className="text-primary">
        {/* The innermost arc and the point under it are the only accented marks
            — the aperture has a centre, and that is where the cursor is going. */}
        <path
          d={`M ${MID_X - RINGS[0]} ${MID_Y} A ${RINGS[0]} ${RINGS[0]} 0 0 1 ${MID_X + RINGS[0]} ${MID_Y}`}
          stroke="currentColor"
          strokeOpacity={0.55}
          strokeWidth={1.2}
          fill="none"
        />
        <circle
          cx={MID_X}
          cy={MID_Y}
          r={2.4}
          fill="currentColor"
          fillOpacity={0.9}
        />
      </g>
    </svg>
  )
}
