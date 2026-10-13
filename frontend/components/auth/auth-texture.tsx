import { useId } from "react"

const COLUMNS = 36
const ROWS = 12
const CELL_WIDTH = 20
const CELL_HEIGHT = 24
const GLYPHS = ".:;+*={}[]<>/01"
const WORDS = [
  "slate",
  "token",
  "context",
  "ask",
  "think",
  "answer",
  "listen",
  "reason",
  "search",
  "write",
  "learn",
  "assist",
]
const WORD_EVERY = 7
const WORD_CELLS = 4
const LINES = Array.from({ length: ROWS }, (_, row) => {
  const tokens: { x: number; text: string }[] = []
  for (let col = 0; col < COLUMNS; col++) {
    const hash = Math.imul(col + 1, 374761393) ^ Math.imul(row + 1, 668265263)
    const value = Math.imul(hash ^ (hash >>> 13), 1274126177) >>> 0
    const word = value % WORD_EVERY === 0 && col + WORD_CELLS <= COLUMNS
    tokens.push({
      x: col * CELL_WIDTH,
      text: word ? WORDS[value % WORDS.length] : GLYPHS[value % GLYPHS.length],
    })
    if (word) col += WORD_CELLS - 1
  }
  return tokens
})

export function AuthTexture() {
  const patternId = useId()

  return (
    <svg
      className="auth-glyphs h-full w-full"
      aria-hidden="true"
      focusable="false"
    >
      <defs>
        <pattern
          id={patternId}
          width={COLUMNS * CELL_WIDTH}
          height={ROWS * CELL_HEIGHT}
          patternUnits="userSpaceOnUse"
        >
          {LINES.map((tokens, row) => (
            <text key={row} y={row * CELL_HEIGHT + CELL_HEIGHT / 2}>
              {tokens.map(({ x, text }) => (
                <tspan key={x} x={x}>
                  {text}
                </tspan>
              ))}
            </text>
          ))}
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill={`url(#${patternId})`} />
    </svg>
  )
}
