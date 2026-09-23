import type { CSSProperties } from "react"

import "./figure-trellis.css"

import { cn, quantize } from "@/lib/utils"

const WIDTH = 400
const HEIGHT = 120
const SPACING = 22
const ROW_HEIGHT = (SPACING * Math.sqrt(3)) / 2
const CENTRE_Y = 56
const PASS_SECONDS = 12
const GROW_SECONDS = 0.8
// Open on a thought already taking shape, rather than an empty field.
const OPENING_SECONDS = 4

type Cell = readonly [q: number, r: number]

const point = ([q, r]: Cell) => ({
  x: WIDTH / 2 + SPACING * (q + r / 2),
  y: quantize(CENTRE_Y + ROW_HEIGHT * r),
})

const FIELD = Array.from({ length: 5 }, (_, row) =>
  Array.from({ length: 13 }, (_, col) => {
    const cell: Cell = [col - 6 - Math.floor((row - 2) / 2), row - 2]
    const position = point(cell)
    const distance = Math.abs(position.x - WIDTH / 2) / (SPACING * 7)
    return {
      ...position,
      opacity: quantize(
        0.08 + 0.25 * (1 - distance) ** 1.6 * (1 - Math.abs(row - 2) * 0.16)
      ),
    }
  })
).flat()

// Parent-first trees on the same six-direction lattice as the throbber.
// Authored silhouettes keep this larger, decorative mark sparse and balanced.
const TREES: readonly (readonly Cell[])[] = [
  [
    [0, 0],
    [1, -1],
    [2, -1],
    [3, -2],
    [4, -2],
    [2, 0],
    [3, 0],
    [-1, 0],
    [-2, 1],
    [-3, 1],
    [-3, 2],
  ],
  [
    [0, 0],
    [0, 1],
    [1, 1],
    [2, 0],
    [3, -1],
    [4, -1],
    [-1, 2],
    [-2, 2],
    [0, -1],
    [-1, -1],
    [-1, -2],
  ],
  [
    [0, 0],
    [-1, 0],
    [-1, -1],
    [-2, -1],
    [-2, -2],
    [-2, 1],
    [-3, 2],
    [1, 0],
    [2, -1],
    [3, -1],
    [1, 1],
  ],
]

const PARENTS = [
  [0, 0, 1, 2, 3, 2, 5, 0, 7, 8, 9],
  [0, 0, 1, 2, 3, 4, 1, 6, 0, 8, 9],
  [0, 0, 1, 2, 3, 1, 5, 0, 7, 8, 7],
]

const BRANCHES = TREES.map((cells, tree) => {
  const depths = [0]
  return cells.slice(1).map((cell, index) => {
    const child = index + 1
    const parent = PARENTS[tree][child]
    const depth = depths[parent] + 1
    depths[child] = depth
    return { from: point(cells[parent]), to: point(cell), depth }
  })
})

export interface FigureTrellisProps {
  className?: string
}

export function FigureTrellis({ className }: FigureTrellisProps) {
  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      fill="none"
      aria-hidden="true"
      focusable="false"
      className={cn("h-auto w-full", className)}
      style={
        {
          "--trellis-cycle": `${PASS_SECONDS * TREES.length}s`,
        } as CSSProperties
      }
    >
      <g fill="var(--throbber-base)">
        {FIELD.map(({ x, y, opacity }) => (
          <circle key={`${x},${y}`} cx={x} cy={y} r={1.15} opacity={opacity} />
        ))}
      </g>
      {BRANCHES.map((branches, tree) => (
        <g key={tree} className="trellis-thought" data-still={tree === 0}>
          {branches.map(({ from, to, depth }, index) => (
            <g
              key={index}
              style={
                {
                  "--trellis-delay": `${tree * PASS_SECONDS + (depth - 1) * GROW_SECONDS - PASS_SECONDS * TREES.length - OPENING_SECONDS}s`,
                  "--trellis-ink": quantize(0.72 - depth * 0.065),
                } as CSSProperties
              }
            >
              <line
                className="trellis-branch"
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                pathLength={1}
                stroke="var(--throbber-spark)"
                strokeWidth={1}
                strokeLinecap="round"
              />
              <circle
                className="trellis-tip"
                cx={to.x}
                cy={to.y}
                r={1.65}
                fill="var(--throbber-spark)"
              />
            </g>
          ))}
        </g>
      ))}
      <circle
        cx={WIDTH / 2}
        cy={CENTRE_Y}
        r={2}
        fill="var(--throbber-spark)"
        opacity={0.65}
      />
    </svg>
  )
}
