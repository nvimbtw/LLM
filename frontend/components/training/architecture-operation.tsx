import type { Operation } from "@/lib/architecture-flow"

export interface ArchitectureOperationProps {
  operation: Operation
  progress: number | null
}

/** Draws a schematic transformation inside an operation. Null progress keeps a legible static example; active fractions express processing rather than a passing particle. */
export function ArchitectureOperation({
  operation,
  progress,
}: ArchitectureOperationProps) {
  const t = progress ?? (operation === "sample" ? 1 : 0.5)
  const active = progress !== null
  const wave = Math.sin(t * Math.PI)
  return (
    <g
      className={active ? "text-primary" : "text-muted-foreground"}
      fill="currentColor"
      stroke="currentColor"
      strokeWidth="1"
      opacity={active ? 0.9 : 0.55}
    >
      {operation === "embedding" &&
        [0, 1, 2, 3, 4, 5, 6].map((index) => {
          const height = 4 + t * (6 + ((index * 7) % 12))
          return (
            <rect
              key={index}
              x={index * 10 + 5}
              y={10 - height / 2}
              width={7 - t * 3}
              height={height}
              stroke="none"
            />
          )
        })}
      {operation === "attention" &&
        [0, 1, 2, 3].flatMap((row) =>
          [0, 1, 2, 3].map((column) => (
            <rect
              key={`${row}-${column}`}
              x={18 + column * 11}
              y={row * 5}
              width="8"
              height="3"
              stroke="none"
              opacity={
                column > row
                  ? 0.08
                  : active
                    ? 0.15 +
                      0.85 * Math.max(0, 1 - Math.abs(t * 5 - row - column / 4))
                    : 0.3 + (row === column ? 0.5 : 0)
              }
            />
          ))
        )}
      {operation === "feedforward" &&
        [0, 1, 2, 3, 4].map((index) => (
          <path
            key={index}
            d={`M4 ${6 + index * 2} L38 ${10 + (index - 2) * (2 + wave * 3)} L76 ${6 + index * 2}`}
            fill="none"
            opacity={0.35 + wave * 0.5}
          />
        ))}
      {operation === "norm" &&
        [0, 1, 2, 3, 4, 5, 6].map((index) => {
          const height = (5 + ((index * 7) % 15)) * (1 - t) + 12 * t
          return (
            <path
              key={index}
              d={`M${10 + index * 10} ${10 - height / 2}v${height}`}
              strokeWidth="3"
            />
          )
        })}
      {operation === "output" &&
        [0, 1, 2, 3, 4, 5, 6].map((index) => {
          const height = 3 + [5, 9, 3, 16, 7, 2, 4][index] * t
          return (
            <rect
              key={index}
              x={6 + index * 10}
              y={20 - height}
              width="6"
              height={height}
              stroke="none"
            />
          )
        })}
      {operation === "sample" &&
        [0, 1, 2].map((index) => (
          <rect
            key={index}
            x={4 + index * 26}
            y="4"
            width={index === 1 ? 24 : 16}
            height="12"
            stroke="none"
            opacity={index === 1 ? 0.45 + t * 0.55 : 0.6 * (1 - t)}
          />
        ))}
      {operation === "loss" && (
        <>
          <path d="M6 2L36 10L6 18 M36 10H72" fill="none" opacity="0.4" />
          <circle cx={6 + 30 * t} cy={2 + 8 * t} r="2" stroke="none" />
          <circle cx={6 + 30 * t} cy={18 - 8 * t} r="2" stroke="none" />
          <rect
            x="64"
            y={10 - (3 + t * 4)}
            width="5"
            height={6 + t * 8}
            stroke="none"
          />
        </>
      )}
      {operation === "backward" &&
        [-1, 0, 1].map((branch) => (
          <g key={branch}>
            <path
              d={`M74 10H48L8 ${10 + branch * 8}`}
              fill="none"
              opacity="0.4"
            />
            <circle
              cx={74 - 66 * t}
              cy={10 + branch * 8 * t}
              r="2"
              stroke="none"
            />
          </g>
        ))}
      {operation === "optimizer" &&
        [0, 1, 2, 3, 4, 5, 6].map((index) => (
          <g key={index}>
            <path d={`M${10 + index * 10} 2v16`} opacity="0.25" />
            <rect
              x={7 + index * 10}
              y={3 + ((index * 3) % 10) * (1 - t) + ((index * 7) % 10) * t}
              width="6"
              height="3"
              stroke="none"
            />
          </g>
        ))}
    </g>
  )
}
