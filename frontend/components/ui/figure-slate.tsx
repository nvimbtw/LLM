import "./figure-slate.css"

import { cn } from "@/lib/utils"

const COLUMNS = 144
const MOUNTAIN_ROWS = 26
const WORD_GAP_ROWS = 2
const CELL_WIDTH = 40 / 9
const CELL_HEIGHT = 6.5
const WIDTH = COLUMNS * CELL_WIDTH
const LETTER_TOP = MOUNTAIN_ROWS + WORD_GAP_ROWS
const LETTER_GAP = 3
const DENSITY = " .:coO08@"

type Point = readonly [number, number]
type Face = readonly [Point, Point, Point, number]

const SHIMMER_CLUSTER_SIZE = 12
const SHIMMER_STEP_SECONDS = 4
// The order alternates across the range rather than tracing a moving scan line.
const SHIMMER_ANCHORS: readonly Point[] = [
  [79, 6],
  [40, 11],
  [110, 14],
  [53, 10],
  [83, 10],
  [73, 10],
]

// Shared vertices make a continuous massif. Light is authored per face so the
// same character grid describes both the silhouette and the folds inside it.
const SUMMIT: Point = [79, 1]
const SHOULDER: Point = [68, 10]
const SPUR: Point = [83, 11]
const FOOT: Point = [60, 25]
const FACES: readonly Face[] = [
  [[2, 25], [24, 18], [34, 26], 0.22],
  [[24, 18], [41, 8], [34, 26], 0.62],
  [[41, 8], [45, 12], [34, 26], 0.43],
  [[41, 8], [45, 12], [48, 10], 0.21],
  [[34, 26], [45, 12], [51, 26], 0.17],
  [[45, 12], [54, 6], [51, 26], 0.67],
  [[54, 6], [59, 11], [51, 26], 0.39],
  [[59, 11], [64, 13], [51, 26], 0.18],
  [[51, 26], [64, 13], FOOT, 0.24],
  [FOOT, [64, 13], SHOULDER, 0.46],
  [FOOT, SHOULDER, SUMMIT, 0.77],
  [FOOT, SUMMIT, SPUR, 0.52],
  [SUMMIT, [85, 6], SPUR, 0.27],
  [[85, 6], [89, 7], SPUR, 0.46],
  [[89, 7], [94, 13], SPUR, 0.23],
  [FOOT, SPUR, [77, 26], 0.3],
  [SPUR, [94, 13], [99, 26], 0.15],
  [SPUR, [77, 26], [99, 26], 0.2],
  [[94, 13], [100, 15], [99, 26], 0.36],
  [[100, 15], [113, 9], [99, 26], 0.58],
  [[113, 9], [115, 14], [99, 26], 0.4],
  [[113, 9], [122, 18], [123, 26], 0.2],
  [[113, 9], [99, 26], [123, 26], 0.28],
  [[122, 18], [129, 16], [123, 26], 0.43],
  [[129, 16], [143, 25], [123, 26], 0.22],
]

const DISTANT_RIDGE: readonly Point[] = [
  [0, 19],
  [9, 15],
  [14, 16],
  [25, 8],
  [31, 12],
  [35, 11],
  [49, 20],
  [63, 14],
  [77, 18],
  [97, 9],
  [102, 12],
  [115, 5],
  [120, 8],
  [124, 7],
  [134, 14],
  [144, 17],
]

const LETTERS = [
  [
    "..#########..",
    ".###########.",
    "###.......###",
    "###..........",
    ".#########...",
    "...#########.",
    "..........###",
    "###.......###",
    ".###########.",
    "..#########..",
  ],
  [
    "###.......",
    "###.......",
    "###.......",
    "###.......",
    "###.......",
    "###.......",
    "###.......",
    "###.......",
    "##########",
    "##########",
  ],
  [
    "....#####....",
    "...#######...",
    "..###...###..",
    ".###.....###.",
    "###.......###",
    "#############",
    "#############",
    "###.......###",
    "###.......###",
    "###.......###",
  ],
  [
    "#############",
    "#############",
    ".....###.....",
    ".....###.....",
    ".....###.....",
    ".....###.....",
    ".....###.....",
    ".....###.....",
    ".....###.....",
    ".....###.....",
  ],
  [
    "###########",
    "###########",
    "###........",
    "###........",
    "#########..",
    "#########..",
    "###........",
    "###........",
    "###########",
    "###########",
  ],
] as const

const ROWS = LETTER_TOP + LETTERS[0].length
const HEIGHT = ROWS * CELL_HEIGHT
export const SLATE_GRID = {
  width: WIDTH,
  cellWidth: CELL_WIDTH,
  cellHeight: CELL_HEIGHT,
}
const WORD_WIDTH = LETTERS.reduce(
  (width, letter) => width + letter[0].length,
  (LETTERS.length - 1) * LETTER_GAP
)
const WORD_LEFT = Math.floor((COLUMNS - WORD_WIDTH) / 2)

function noise(col: number, row: number) {
  const hash = Math.imul(col + 1, 374761393) ^ Math.imul(row + 1, 668265263)
  return ((Math.imul(hash ^ (hash >>> 13), 1274126177) >>> 0) % 1024) / 1024
}

function faceLight(x: number, y: number, [a, b, c, light]: Face) {
  const area = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
  const u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / area
  const v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / area
  if (u < 0 || v < 0 || u + v > 1) return 0

  const strata = Math.sin(x * 1.7 + y * 2.1) * Math.sin(x * 0.3 - y * 0.8)
  const ridge = Math.min(u, v, 1 - u - v) < 0.035 ? 0.1 : 0
  const falloff = Math.max(0, (y - 15) / 13)
  return Math.max(0.03, light + strata * 0.17 + ridge - falloff * 0.28)
}

function distantLight(x: number, y: number) {
  for (let i = 1; i < DISTANT_RIDGE.length; i++) {
    const [left, right] = [DISTANT_RIDGE[i - 1], DISTANT_RIDGE[i]]
    if (x < left[0] || x >= right[0]) continue
    const horizon =
      left[1] + ((right[1] - left[1]) * (x - left[0])) / (right[0] - left[0])
    if (y < horizon || y > 25) return 0
    return 0.12 + Math.max(0, 1 - (y - horizon) / 9) * 0.16
  }
  return 0
}

function letterAt(col: number, row: number) {
  const x = col - WORD_LEFT
  const y = row - LETTER_TOP
  if (x < 0 || x >= WORD_WIDTH || y < 0 || y >= LETTERS[0].length) return false
  let left = 0
  for (const letter of LETTERS) {
    if (x >= left && x < left + letter[0].length) {
      return letter[y][x - left] === "#"
    }
    left += letter[0].length + LETTER_GAP
  }
  return false
}

function sample(col: number, row: number) {
  const grain = noise(col, row)
  if (letterAt(col, row)) {
    return { glyph: "08@@80@8"[(col + row * 3) % 8], ink: grain > 0.87 ? 6 : 7 }
  }
  if (row >= MOUNTAIN_ROWS) return { glyph: " ", ink: 0 }

  const face = FACES.reduce(
    (light, facet) => faceLight(col, row, facet) || light,
    0
  )
  const light = face || distantLight(col, row)
  const edge = Math.min(1, col / 10, (COLUMNS - 1 - col) / 10)
  const mist = Math.max(0, 1 - Math.abs(row - 22) / 5) * 0.08
  const value = (light + mist) * edge + (grain - 0.5) * 0.09
  if (value < 0.055) {
    return { glyph: " ", ink: 0 }
  }
  const density = Math.min(
    DENSITY.length - 1,
    Math.max(1, Math.floor(value * 10))
  )
  return {
    glyph: DENSITY[density],
    ink: Math.min(5, Math.max(0, Math.floor(value * 7))),
  }
}

const GRID = Array.from({ length: ROWS }, (_, row) =>
  Array.from({ length: COLUMNS }, (_, col) => ({
    col,
    row,
    ...sample(col, row),
  }))
)

// Reserve occupied cells in the page texture so its glyphs never overprint the
// authored mountains or lettering. Each path joins adjacent occupied cells.
export const SLATE_CUTOUTS = GRID.map((cells, row) => {
  let start = -1
  const runs: string[] = []
  for (let col = 0; col <= COLUMNS; col++) {
    if (col < COLUMNS && cells[col].glyph !== " ") {
      if (start < 0) start = col
    } else if (start >= 0) {
      runs.push(
        `M${start * CELL_WIDTH},${(row + 0.5) * CELL_HEIGHT}h${(col - start) * CELL_WIDTH}`
      )
      start = -1
    }
  }
  return runs.join(" ")
})

function shimmerClusters() {
  const ridges = GRID.flat().filter(({ ink }) => ink === 4 || ink === 5)
  const clusters = new Map<number, number>()
  SHIMMER_ANCHORS.forEach(([x, y], cluster) => {
    const nearby = ridges
      .filter(({ col, row }) => !clusters.has(row * COLUMNS + col))
      .map((cell) => ({
        ...cell,
        distance: Math.hypot(
          (cell.col - x) * CELL_WIDTH,
          (cell.row - y) * CELL_HEIGHT
        ),
      }))
      .sort((a, b) => a.distance - b.distance || a.row - b.row || a.col - b.col)
      .slice(0, SHIMMER_CLUSTER_SIZE)
    nearby.forEach(({ col, row }) => clusters.set(row * COLUMNS + col, cluster))
  })
  return clusters
}

const CLUSTERS = shimmerClusters()

// Each glyph belongs to exactly one run. Splitting only at a change of ink or
// shimmer phase keeps the DOM small and leaves every character on its grid.
const LINES = GRID.flatMap((cells, row) => {
  const runs = new Map<
    string,
    {
      ink: number
      cluster: number
      columns: number[]
      text: string
    }
  >()
  cells.forEach(({ col, ink, glyph }) => {
    if (glyph === " ") return
    const cluster = CLUSTERS.get(row * COLUMNS + col) ?? -1
    const key = `${ink}:${cluster}`
    const run = runs.get(key) ?? { ink, cluster, columns: [], text: "" }
    run.columns.push(col)
    run.text += glyph
    runs.set(key, run)
  })
  return Array.from(runs.values(), ({ columns, ...run }) => ({
    ...run,
    row,
    x: columns.map((col) => (col * CELL_WIDTH).toFixed(3)).join(" "),
  }))
})

export interface FigureSlateProps {
  className?: string
  part?: "all" | "word"
  shimmer?: boolean
}

export function FigureSlate({
  className,
  part = "all",
  shimmer = true,
}: FigureSlateProps) {
  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      aria-hidden="true"
      focusable="false"
      className={cn("slate-mark h-auto w-full", className)}
    >
      {LINES.filter(({ row }) => part === "all" || row >= LETTER_TOP).map(
        ({ row, ink, cluster, x, text }) => (
          <text
            key={`${row}:${ink}:${cluster}`}
            x={x}
            y={(row + 1) * CELL_HEIGHT - 1}
            className={cn(
              `slate-mark-ink-${ink}`,
              shimmer && cluster >= 0 && "slate-mark-shimmer"
            )}
            style={
              shimmer && cluster >= 0
                ? { animationDelay: `${cluster * SHIMMER_STEP_SECONDS}s` }
                : undefined
            }
          >
            {text}
          </text>
        )
      )}
    </svg>
  )
}
