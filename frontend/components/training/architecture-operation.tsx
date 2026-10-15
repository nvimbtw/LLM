import {
  diagramPath,
  phaseProgress,
  pointOnRoute,
  type Operation,
} from "@/lib/architecture-flow"

export interface ArchitectureOperationProps {
  operation: Operation
  progress: number
  emphasis: number
}

/** Draws a complete operation from its retained progress and smoothly varying emphasis. Every miniature has a stable finished state; branch marks follow their actual curves. */
export function ArchitectureOperation({
  operation,
  progress: t,
  emphasis,
}: ArchitectureOperationProps) {
  const wave = Math.sin(t * Math.PI)
  return (
    <g
      style={{
        color: `color-mix(in oklab, var(--primary) ${emphasis * 100}%, var(--muted-foreground))`,
      }}
      fill="currentColor"
      stroke="currentColor"
      strokeWidth="1"
      opacity={0.55 + 0.35 * emphasis}
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
                  : 0.12 +
                    (row === column ? 0.88 : 0.5) *
                      phaseProgress(
                        t,
                        row * 0.16 + column * 0.025,
                        row * 0.16 + column * 0.025 + 0.35
                      )
              }
            />
          ))
        )}
      {operation === "feedforward" &&
        [0, 1, 2, 3, 4].map((index) => (
          <path
            key={index}
            d={`M4 ${6 + index * 2} C20 ${6 + index * 2} 24 ${10 + (index - 2) * (2 + wave * 3)} 38 ${10 + (index - 2) * (2 + wave * 3)} S60 ${6 + index * 2} 76 ${6 + index * 2}`}
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
          {[2, 18].map((y) => {
            const route = [
              { x: 6, y },
              { x: 36, y: 10 },
              { x: 68, y: 10 },
            ]
            const point = pointOnRoute(route, t)
            return (
              <g key={y}>
                <path d={diagramPath(route)} fill="none" opacity="0.4" />
                <circle
                  cx={point.x}
                  cy={point.y}
                  r="2"
                  stroke="none"
                  opacity={1 - phaseProgress(t, 0.85, 1)}
                />
              </g>
            )
          })}
          <rect
            x="66"
            y={10 - (3 + phaseProgress(t, 0.5, 1) * 4)}
            width="5"
            height={6 + phaseProgress(t, 0.5, 1) * 8}
            stroke="none"
          />
        </>
      )}
      {operation === "backward" &&
        [-1, 0, 1].map((branch) => {
          const route = [
            { x: 74, y: 10 },
            { x: 48, y: 10 },
            { x: 8, y: 10 + branch * 8 },
          ]
          const point = pointOnRoute(route, t)
          return (
            <g key={branch}>
              <path d={diagramPath(route)} fill="none" opacity="0.4" />
              <circle cx={point.x} cy={point.y} r="2" stroke="none" />
            </g>
          )
        })}
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
