import type { CSSProperties } from "react"

import { cn, quantize } from "@/lib/utils"

/**
 * A rule across the page, opened at the centre.
 *
 * The new-chat figure. Deliberately the quietest of the three: it sits directly
 * above a composer that has to stay the subject of the screen, so it is one line
 * and an aperture in it, and it fades out at both ends rather than stopping —
 * a hard stop would read as a divider, which is a different thing entirely.
 *
 * Pure and server-rendered, same terms as `figure-lattice.tsx` and
 * `figure-seed.tsx`. The fade is stepped segments rather than a
 * `<linearGradient>` on purpose: a gradient needs an `id`, and an `id` in a
 * component is a collision waiting for the second instance on a page.
 *
 * It idles. The rings swell outward from the aperture on a 14s cycle and the
 * centre point pulses under them — the whole thing is opacity, driven by CSS
 * keyframes in `globals.css` (`.horizon-ring` / `.horizon-core`), so there is
 * still no client component and no JavaScript behind it. Each ring animates
 * around its own authored opacity, passed down as `--o`, which is what keeps
 * the fade from the centre intact while it breathes; `--d` staggers them into a
 * wave. `prefers-reduced-motion` returns the static mark.
 */

const WIDTH = 400
const HEIGHT = 120
const MID_X = WIDTH / 2
const MID_Y = 66
const GAP = 54
const SEGMENTS = 7
const RINGS = [15, 25, 36, 48]

// Seconds between one ring's swell and the next one out. Small against the 14s
// cycle: the wave should read as one movement passing through the figure, not
// as four rings taking turns.
const DELAY_STEP = 0.6

// From the aperture outward to the edge, in equal steps. Opacity falls with
// distance, so the line arrives out of the paper instead of starting on it.
const SEGMENT_SPAN = (MID_X - GAP) / SEGMENTS

const RULE = Array.from({ length: SEGMENTS }, (_, i) => ({
  inner: GAP + i * SEGMENT_SPAN,
  outer: GAP + (i + 1) * SEGMENT_SPAN,
  opacity: quantize(0.3 * (1 - i / SEGMENTS) ** 1.3),
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
            className="horizon-ring"
            // The base opacity travels as a variable rather than as the
            // attribute, because the keyframes multiply it — set both and the
            // animation would win and flatten the four rings to one value.
            style={
              {
                "--o": quantize(0.26 * (1 - i / RINGS.length) ** 0.9),
                "--d": `${i * DELAY_STEP}s`,
              } as CSSProperties
            }
            d={`M ${MID_X - r} ${MID_Y} A ${r} ${r} 0 0 1 ${MID_X + r} ${MID_Y}`}
          />
        ))}
      </g>

      <g className="text-primary">
        {/* The innermost arc and the point under it are the only accented marks
            — the aperture has a centre, and that is where the cursor is going.
            The arc leads the wave: it is ring zero, so it carries no delay. */}
        <path
          className="horizon-ring"
          style={{ "--o": 0.55 } as CSSProperties}
          d={`M ${MID_X - RINGS[0]} ${MID_Y} A ${RINGS[0]} ${RINGS[0]} 0 0 1 ${MID_X + RINGS[0]} ${MID_Y}`}
          stroke="currentColor"
          strokeWidth={1.2}
          fill="none"
        />
        <circle
          className="horizon-core"
          cx={MID_X}
          cy={MID_Y}
          r={2.4}
          fill="currentColor"
        />
      </g>
    </svg>
  )
}
