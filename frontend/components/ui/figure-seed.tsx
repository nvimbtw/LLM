import { cn } from "@/lib/utils"

/**
 * One filled node on an empty lattice, and what grows out of it.
 *
 * The signup screen's figure, and the counterpart to `figure-lattice.tsx` on
 * login: same hairline vocabulary, opposite message. Login is a sentence already
 * being read; this is a grid of positions with exactly one of them taken and a
 * few steps of reach into the rest.
 *
 * Static and pure, on the same terms as the lattice: fixed constants, module
 * scope, `currentColor` for both themes. The depth ramp is computed here rather
 * than annotated by hand on each edge, so adding an edge cannot put an edge and
 * its fade out of step.
 */

const WIDTH = 320
const HEIGHT = 200
const COLS = 9
const ROWS = 6
const X0 = 28
const Y0 = 24
const DX = 33
const DY = 30

const nodeX = (col: number) => X0 + col * DX
const nodeY = (row: number) => Y0 + row * DY

const ORIGIN = "1,3"

/**
 * The tree, as `"parent → child"` cell keys — one edge per line.
 *
 * Written as strings rather than nested `[[col,row],[col,row]]` tuples because
 * the tuple form is what the figure *is*, and the formatter unrolls it into
 * ninety lines you cannot see the shape in. A key is the node's identity
 * everywhere else here anyway, so this is the form with no conversion in it.
 *
 * Listed parent-first, which is what lets `DEPTH` settle in a single pass.
 */
const EDGES = [
  "1,3 2,3",
  "2,3 3,2",
  "2,3 3,4",
  "3,2 4,1",
  "3,2 4,2",
  "3,4 4,4",
  "4,1 5,0",
  "4,2 5,2",
  "4,4 5,5",
  "5,2 6,1",
  "5,2 6,3",
  "5,5 6,5",
  "6,1 7,0",
  "6,3 7,3",
  "7,3 8,2",
].map((edge) => {
  const [from, to] = edge.split(" ")
  return { from, to }
})

const coords = (key: string) => {
  const [col, row] = key.split(",").map(Number)
  return { x: nodeX(col), y: nodeY(row) }
}

// Breadth-first from the origin. Depth is what every fade in the figure is a
// function of: an edge, its far node, and that node's halo all dim together, so
// the growth reads as spending itself rather than as three separate gradients.
const DEPTH = (() => {
  const depth = new Map<string, number>([[ORIGIN, 0]])
  for (const { from, to } of EDGES) {
    const parent = depth.get(from)
    if (parent !== undefined) depth.set(to, parent + 1)
  }
  return depth
})()

const MAX_DEPTH = Math.max(...DEPTH.values())

// 1 at the origin, easing to ~0.2 at the frontier. Not linear: the first step
// out should still look decisively lit, and only the last two should trail off.
const reach = (d: number) => 1 - 0.8 * (d / MAX_DEPTH) ** 1.4

// Every cell in the lattice, tagged with its depth if the tree reached it.
const CELLS = Array.from({ length: COLS * ROWS }, (_, i) => {
  const col = i % COLS
  const row = Math.floor(i / COLS)
  const key = `${col},${row}`
  return { key, x: nodeX(col), y: nodeY(row), depth: DEPTH.get(key) }
})

const ORIGIN_POINT = coords(ORIGIN)

export interface FigureSeedProps {
  className?: string
}

export function FigureSeed({ className }: FigureSeedProps) {
  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      fill="none"
      aria-hidden="true"
      focusable="false"
      className={cn("h-auto w-full text-foreground", className)}
    >
      {/* The unreached lattice. Faint enough to be a field rather than a set of
          marks — its job is to say the space is larger than what is on it. */}
      <g fill="currentColor" fillOpacity={0.16}>
        {CELLS.filter((c) => c.depth === undefined).map((c) => (
          <circle key={c.key} cx={c.x} cy={c.y} r={1.4} />
        ))}
      </g>

      <g className="text-primary">
        <g stroke="currentColor" strokeWidth={1.1} strokeLinecap="round">
          {EDGES.map(({ from, to }) => {
            const a = coords(from)
            const b = coords(to)
            return (
              <line
                key={`${from}-${to}`}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                strokeOpacity={0.55 * reach(DEPTH.get(to) ?? MAX_DEPTH)}
              />
            )
          })}
        </g>

        <g fill="currentColor">
          {CELLS.filter((c) => c.depth !== undefined).map((c) => (
            <circle
              key={c.key}
              cx={c.x}
              cy={c.y}
              r={c.depth === 0 ? 3.4 : 2.1}
              fillOpacity={0.9 * reach(c.depth ?? 0)}
            />
          ))}
        </g>

        {/* The halo marks the one cell that was already yours. Two rings rather
            than one so it reads as an origin and not as a slightly larger dot. */}
        <circle
          cx={ORIGIN_POINT.x}
          cy={ORIGIN_POINT.y}
          r={7.5}
          stroke="currentColor"
          strokeOpacity={0.45}
          strokeWidth={1}
        />
        <circle
          cx={ORIGIN_POINT.x}
          cy={ORIGIN_POINT.y}
          r={13}
          stroke="currentColor"
          strokeOpacity={0.18}
          strokeWidth={1}
        />
      </g>
    </svg>
  )
}
