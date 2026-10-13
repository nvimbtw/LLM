import { useId } from "react"

import { SLATE_CUTOUTS, SLATE_GRID } from "@/components/ui/figure-slate"

const COLUMNS = 48
const ROWS = 24
const WIDTH = COLUMNS * SLATE_GRID.cellWidth
const HEIGHT = ROWS * SLATE_GRID.cellHeight
const GLYPHS = ".:coO08@"
const LINES = Array.from({ length: ROWS }, (_, row) =>
  Array.from({ length: COLUMNS }, (_, col) => {
    const hash = Math.imul(col + 1, 374761393) ^ Math.imul(row + 1, 668265263)
    return GLYPHS[
      (Math.imul(hash ^ (hash >>> 13), 1274126177) >>> 0) % GLYPHS.length
    ]
  }).join("")
)
const POSITIONS = Array.from({ length: COLUMNS }, (_, col) =>
  (col * SLATE_GRID.cellWidth).toFixed(3)
).join(" ")

export interface SlateTextureProps {
  x: number
  y: number
  scale: number
}

export function SlateTexture({ x, y, scale }: SlateTextureProps) {
  const id = useId()
  const maskId = `${id}-clear`

  return (
    <svg
      className="slate-mark h-full w-full"
      aria-hidden="true"
      focusable="false"
    >
      <defs>
        <mask id={maskId}>
          <rect width="100%" height="100%" fill="white" />
          <g transform={`translate(${x} ${y}) scale(${scale})`}>
            {SLATE_CUTOUTS.map((d, row) => (
              <path
                key={row}
                d={d}
                stroke="black"
                strokeWidth={SLATE_GRID.cellHeight}
              />
            ))}
          </g>
        </mask>
        <pattern
          id={id}
          x={x}
          y={y}
          width={WIDTH * scale}
          height={HEIGHT * scale}
          patternUnits="userSpaceOnUse"
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        >
          {LINES.map((text, row) => (
            <text
              key={row}
              x={POSITIONS}
              y={(row + 1) * SLATE_GRID.cellHeight - 1}
            >
              {text}
            </text>
          ))}
        </pattern>
      </defs>
      <rect
        width="100%"
        height="100%"
        fill={`url(#${id})`}
        mask={`url(#${maskId})`}
      />
    </svg>
  )
}
