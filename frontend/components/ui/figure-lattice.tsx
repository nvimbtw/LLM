import { cn, quantize } from "@/lib/utils"

/**
 * A sentence, tokenized, with a few of the long-range links drawn in.
 *
 * The login screen's figure. It is a *diagram* rather than an ornament — ticks
 * on a baseline are the shape of a token stream, and the arcs over the top are
 * the one thing about the model worth putting on a login page: that a token near
 * the end is conditioned on one near the beginning.
 *
 * Static and pure. Every coordinate is computed once at module scope from fixed
 * constants, so the server and the client draw the identical path — no
 * `Math.random`, no clock, nothing to hydrate. It carries no colour of its own
 * either: strokes are `currentColor` at graded opacities, and the accent group
 * re-points `currentColor` at `--primary`. Both themes are handled by the tokens
 * the parent already sets, which is why there is no dark-mode variant.
 */

const WIDTH = 320
const HEIGHT = 200
const BASELINE = 150
const TICKS = 26
const TICK_X0 = 26
const TICK_GAP = 10.7

const tickX = (i: number) => TICK_X0 + i * TICK_GAP

// Varied, but deterministic and not *random*-looking: two incommensurate sines
// give a run of heights that never settles into a pattern the eye can finish,
// which is what a real distribution of token lengths looks like.
const TICK_HEIGHT = Array.from({ length: TICKS }, (_, i) => {
  const wave = Math.sin(i * 1.7 + 0.6) * 0.6 + Math.cos(i * 0.53) * 0.4
  // Quantised because this becomes a coordinate attribute: `Math.sin`/`Math.cos`
  // are implementation-defined to the last bit, and Node and the browser
  // disagreeing there is a hydration mismatch on a figure nobody touched.
  return quantize(7 + 21 * (0.5 + 0.5 * wave))
})

// The ticks drawn in the accent colour — the "attended" ones, i.e. the endpoints
// the arcs below land on. Kept as a Set so the two lists cannot drift apart.
const ARCS: [number, number][] = [
  [1, 8],
  [4, 16],
  [10, 20],
  [13, 25],
  [8, 12],
  [16, 22],
]

const ACCENT = new Set(ARCS.flat())

// Quadratic, not elliptical: the control point sits at twice the height, so the
// peak of the curve is exactly `rise` above the baseline and the arcs stack in a
// legible order instead of whatever an `A` command's radii happen to produce.
function arcPath([from, to]: [number, number]) {
  const x1 = tickX(from)
  const x2 = tickX(to)
  const rise = 16 + (x2 - x1) * 0.42
  return `M ${x1} ${BASELINE} Q ${(x1 + x2) / 2} ${BASELINE - rise * 2} ${x2} ${BASELINE}`
}

export interface FigureLatticeProps {
  className?: string
}

export function FigureLattice({ className }: FigureLatticeProps) {
  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      fill="none"
      aria-hidden="true"
      focusable="false"
      className={cn("h-auto w-full text-foreground", className)}
    >
      {/* The arcs sit under the ticks so a tick reads as the thing being linked
          rather than as a break in the line. */}
      <g stroke="currentColor" strokeWidth={0.9} strokeLinecap="round">
        {ARCS.map((arc, i) => (
          <path
            key={i}
            d={arcPath(arc)}
            strokeOpacity={0.13 + (i % 3) * 0.05}
          />
        ))}
      </g>

      <line
        x1={16}
        y1={BASELINE}
        x2={WIDTH - 16}
        y2={BASELINE}
        stroke="currentColor"
        strokeOpacity={0.28}
        strokeWidth={1}
      />

      <g strokeWidth={1.5} strokeLinecap="round">
        {TICK_HEIGHT.map((h, i) => {
          const accent = ACCENT.has(i)
          return (
            <line
              key={i}
              x1={tickX(i)}
              y1={BASELINE}
              x2={tickX(i)}
              y2={BASELINE - h}
              stroke="currentColor"
              className={accent ? "text-primary" : undefined}
              strokeOpacity={accent ? 0.85 : 0.24}
            />
          )
        })}
      </g>

      {/* One dot per linked tick, at its tip — the only filled marks in the
          figure, so the eye lands on the endpoints and reads the arcs from
          them. */}
      <g className="text-primary" fill="currentColor">
        {[...ACCENT].map((i) => (
          <circle
            key={i}
            cx={tickX(i)}
            cy={BASELINE - TICK_HEIGHT[i]}
            r={1.9}
            fillOpacity={0.9}
          />
        ))}
      </g>
    </svg>
  )
}
