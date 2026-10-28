import { useId } from "react"

import { cn } from "@/lib/utils"

const COLUMNS = 64
const ROWS = 16
const CELL_WIDTH = 6
const CELL_HEIGHT = 10
const GLYPHS = ".:coO08@"
const POSITIONS = Array.from(
  { length: COLUMNS },
  (_, column) => column * CELL_WIDTH
).join(" ")
const LINES = Array.from({ length: ROWS }, (_, row) =>
  Array.from({ length: COLUMNS }, (_, column) => {
    const hash =
      Math.imul(column + 1, 374761393) ^ Math.imul(row + 1, 668265263)
    return GLYPHS[
      (Math.imul(hash ^ (hash >>> 13), 1274126177) >>> 0) % GLYPHS.length
    ]
  }).join("")
)

export interface SectionTextureProps {
  tone?: "neutral" | "warm"
  className?: string
}

/** Adds a bounded decorative character field to page identity, composer supplements or an empty state; it never intercepts input or exposes text to assistive technology. */
export function SectionTexture({
  tone = "neutral",
  className,
}: SectionTextureProps) {
  const id = `texture-${useId().replaceAll(":", "")}`
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      data-slot="section-texture"
      data-tone={tone}
      className={cn("section-texture", className)}
    >
      <defs>
        <pattern
          id={id}
          width={COLUMNS * CELL_WIDTH}
          height={ROWS * CELL_HEIGHT}
          patternUnits="userSpaceOnUse"
        >
          {LINES.map((line, row) => (
            <text key={row} x={POSITIONS} y={(row + 1) * CELL_HEIGHT - 1}>
              {line}
            </text>
          ))}
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill={`url(#${id})`} />
    </svg>
  )
}
