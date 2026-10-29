import type { ServiceState, ServiceStatus } from "@/lib/service-health"
import { cn } from "@/lib/utils"
import type { SERVICES } from "./service-health"

type Point = readonly [number, number]
type Stroke = { points: readonly Point[]; width: number; endWidth?: number }
type FigureState = ServiceState | "unavailable"

const CELL_WIDTH = 6
const CELL_HEIGHT = 9
const COLUMNS = 128
const ROWS = 24
const GLYPHS = ".:coO08@"

const TRUNK: readonly Stroke[] = [
  {
    points: [
      [25, 175],
      [82, 158],
      [169, 143],
      [255, 132],
      [344, 117],
      [432, 106],
      [518, 95],
      [603, 84],
      [688, 70],
      [810, 41],
    ],
    width: 0.7,
    endWidth: 14,
  },
  {
    points: [
      [56, 164],
      [87, 149],
      [108, 130],
    ],
    width: 2.3,
  },
  {
    points: [
      [193, 140],
      [221, 150],
      [246, 158],
    ],
    width: 2,
  },
  {
    points: [
      [372, 113],
      [400, 124],
      [423, 141],
    ],
    width: 2,
  },
  {
    points: [
      [536, 93],
      [563, 104],
      [582, 119],
    ],
    width: 1.6,
  },
]

const BRANCHES: readonly {
  id: (typeof SERVICES)[number]["id"]
  tip: Point
  strokes: readonly Stroke[]
}[] = [
  {
    id: "llm-engine",
    tip: [112, 25],
    strokes: [
      {
        points: [
          [165, 143],
          [145, 110],
          [126, 77],
          [112, 35],
        ],
        width: 4,
      },
      {
        points: [
          [137, 100],
          [109, 87],
          [89, 65],
        ],
        width: 1.6,
      },
      {
        points: [
          [124, 70],
          [145, 48],
          [150, 27],
        ],
        width: 1.4,
      },
    ],
  },
  {
    id: "data-engine",
    tip: [230, 20],
    strokes: [
      {
        points: [
          [254, 132],
          [246, 100],
          [234, 70],
          [230, 30],
        ],
        width: 3.6,
      },
      {
        points: [
          [240, 88],
          [216, 71],
          [202, 48],
        ],
        width: 1.5,
      },
      {
        points: [
          [234, 65],
          [257, 49],
          [271, 30],
        ],
        width: 1.3,
      },
    ],
  },
  {
    id: "gyms-engine",
    tip: [354, 14],
    strokes: [
      {
        points: [
          [344, 117],
          [354, 90],
          [358, 60],
          [354, 24],
        ],
        width: 3.6,
      },
      {
        points: [
          [356, 80],
          [330, 64],
          [320, 42],
        ],
        width: 1.5,
      },
      {
        points: [
          [357, 64],
          [380, 47],
          [391, 26],
        ],
        width: 1.4,
      },
    ],
  },
  {
    id: "bench-engine",
    tip: [472, 17],
    strokes: [
      {
        points: [
          [432, 106],
          [450, 81],
          [465, 55],
          [472, 27],
        ],
        width: 3.3,
      },
      {
        points: [
          [453, 74],
          [433, 54],
          [429, 35],
        ],
        width: 1.4,
      },
      {
        points: [
          [466, 53],
          [488, 40],
          [505, 23],
        ],
        width: 1.3,
      },
    ],
  },
  {
    id: "just-bash",
    tip: [588, 17],
    strokes: [
      {
        points: [
          [518, 95],
          [548, 76],
          [572, 52],
          [588, 27],
        ],
        width: 3.1,
      },
      {
        points: [
          [552, 72],
          [540, 50],
          [540, 29],
        ],
        width: 1.3,
      },
      {
        points: [
          [571, 52],
          [600, 45],
          [617, 29],
        ],
        width: 1.3,
      },
    ],
  },
  {
    id: "searxng",
    tip: [704, 20],
    strokes: [
      {
        points: [
          [603, 84],
          [644, 69],
          [681, 49],
          [704, 30],
        ],
        width: 2.8,
      },
      {
        points: [
          [650, 66],
          [646, 44],
          [653, 26],
        ],
        width: 1.2,
      },
      {
        points: [
          [682, 48],
          [714, 45],
          [737, 31],
        ],
        width: 1.2,
      },
    ],
  },
]

const INK: Record<FigureState, string> = {
  online: "text-status-success",
  reconnecting: "text-status-warning",
  offline: "text-destructive",
  not_configured: "text-muted-foreground/45",
  unavailable: "text-muted-foreground/35",
}

/** Measure the distance and position of a grid point along a finite branch segment. */
function segmentPosition(x: number, y: number, a: Point, b: Point) {
  const dx = b[0] - a[0]
  const dy = b[1] - a[1]
  const t = Math.max(
    0,
    Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / (dx * dx + dy * dy))
  )
  return {
    distance: Math.hypot(x - a[0] - t * dx, y - a[1] - t * dy),
    fraction: t,
  }
}

/** Measure the nearest authored stroke at one coordinate, retaining the thickness of that stroke. */
function branchInk(x: number, y: number, strokes: readonly Stroke[]): number {
  let ink = 0
  for (const stroke of strokes) {
    for (let index = 1; index < stroke.points.length; index++) {
      const { distance, fraction } = segmentPosition(
        x,
        y,
        stroke.points[index - 1],
        stroke.points[index]
      )
      const width =
        stroke.endWidth === undefined
          ? stroke.width
          : stroke.width +
            ((stroke.endWidth - stroke.width) * (index - 1 + fraction)) /
              (stroke.points.length - 1)
      ink = Math.max(ink, width + 2 - distance)
    }
  }
  return ink
}

/** Measure small leaf clusters at the ends of the branch's side twigs. */
function leafInk(x: number, y: number, strokes: readonly Stroke[]): number {
  let ink = 0
  for (let index = 1; index < strokes.length; index++) {
    const stroke = strokes[index]
    const [tipX, tipY] = stroke.points[stroke.points.length - 1]
    const radius = Math.hypot((x - tipX) / 17, (y - tipY) / 11)
    if (radius < 1) ink = Math.max(ink, (1 - radius) * 4 + 0.8)
  }
  return ink
}

/** Generate stable monospace glyph runs for wood and optional leaves, without timers or random state. */
function rasterize(strokes: readonly Stroke[], gap?: Point, leaves = false) {
  return Array.from({ length: ROWS }, (_, row) => {
    const positions: number[] = []
    let text = ""
    for (let column = 0; column < COLUMNS; column++) {
      const x = column * CELL_WIDTH
      const y = row * CELL_HEIGHT
      if (gap && Math.hypot(x - gap[0], y - gap[1]) < 9) continue
      const ink = Math.max(
        branchInk(x, y, strokes),
        leaves ? leafInk(x, y, strokes) : 0
      )
      if (ink <= 0) continue
      const grain = (Math.imul(column + 1, 37) ^ Math.imul(row + 3, 101)) >>> 0
      const density = Math.min(
        GLYPHS.length - 1,
        Math.max(0, Math.floor(ink / 1.8 + (grain % 3) / 3))
      )
      positions.push(x)
      text += GLYPHS[density]
    }
    return { y: row * CELL_HEIGHT + CELL_HEIGHT, x: positions.join(" "), text }
  }).filter((line) => line.text.length > 0)
}

const TRUNK_LINES = rasterize(TRUNK)
const BRANCH_LINES = BRANCHES.map((branch) =>
  rasterize(branch.strokes, undefined, true)
)
const BROKEN_BRANCH_LINES = BRANCHES.map((branch) =>
  rasterize(branch.strokes, branch.strokes[0].points[2], true)
)

/** Draw a stable group of monospace runs in the colour and geometry chosen by its parent. */
function InkLines({ lines }: { lines: ReturnType<typeof rasterize> }) {
  return lines.map((line) => (
    <text key={line.y} x={line.x} y={line.y}>
      {line.text}
    </text>
  ))
}

/** Render the six observed services as a character-drawn branch in the list's service order. */
export function ServiceTree({
  services,
}: {
  services: ServiceStatus[] | null
}) {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      viewBox="0 0 768 225"
      className="h-auto w-full font-mono text-muted-foreground"
      style={{ fontSize: 10, fontVariantLigatures: "none" }}
    >
      <g fill="currentColor" opacity="0.55">
        <InkLines lines={TRUNK_LINES} />
      </g>
      {BRANCHES.map((branch, index) => {
        const state: FigureState =
          services?.find((service) => service.id === branch.id)?.state ??
          "unavailable"
        return (
          <g key={branch.id} className={INK[state]} fill="currentColor">
            <g
              opacity={state === "offline" ? 0.45 : 0.8}
              className={cn(
                state === "reconnecting" && "motion-safe:animate-pulse"
              )}
            >
              <InkLines
                lines={
                  state === "offline"
                    ? BROKEN_BRANCH_LINES[index]
                    : BRANCH_LINES[index]
                }
              />
            </g>
            {state === "offline" && (
              <path
                d={`M${branch.tip[0] - 5} ${branch.tip[1] + 6}l10 6m-10 0 10-6`}
                stroke="currentColor"
                strokeWidth="1.5"
              />
            )}
            {state === "not_configured" && (
              <circle
                cx={branch.tip[0]}
                cy={branch.tip[1] + 9}
                r="4"
                fill="none"
                stroke="currentColor"
                strokeDasharray="1 3"
              />
            )}
          </g>
        )
      })}
    </svg>
  )
}
