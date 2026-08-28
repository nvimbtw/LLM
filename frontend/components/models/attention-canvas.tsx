"use client"

import * as React from "react"

import type { AttentionCell, AttentionMatrix } from "@/lib/attention"
import { attentionRamp, rampIndex } from "./attention-ramp"

export interface AttentionCanvasProps {
  matrix: AttentionMatrix
  /** The cell under the pointer, owned by the caller so the readout and the
   *  crosshair cannot disagree about which one it is. */
  cell: AttentionCell | null
  onCellChange: (cell: AttentionCell | null) => void
  /** Wall time of the last full redraw. Reported rather than kept private
   *  because whether canvas2d is enough for this figure is a question someone
   *  will have to answer with a number, on their own machine. */
  onDrawn: (milliseconds: number) => void
}

const CHANNELS = 4
const CELL_CENTRE = 0.5
const PERCENT = 100

export function AttentionCanvas({
  matrix,
  cell,
  onCellChange,
  onDrawn,
}: AttentionCanvasProps) {
  const ref = React.useRef<HTMLCanvasElement>(null)
  const { size, weights, tokens } = matrix

  React.useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const context = canvas.getContext("2d", { alpha: true })
    if (!context) return

    const started = performance.now()
    // One backing pixel per cell, scaled to the panel by CSS with
    // nearest-neighbour. A cell is not a shape, it is a pixel — so there is
    // nothing a `fillRect` per cell could buy, and at the 512-token ceiling
    // that would be 262,144 draw calls where this is one `putImageData` and a
    // blit the compositor does.
    canvas.width = size
    canvas.height = size

    const pixels = new Uint8ClampedArray(size * size * CHANNELS)
    for (let i = 0; i < weights.length; i += 1) {
      const from = rampIndex(weights[i])
      const to = i * CHANNELS
      pixels[to] = attentionRamp[from]
      pixels[to + 1] = attentionRamp[from + 1]
      pixels[to + 2] = attentionRamp[from + 2]
      pixels[to + 3] = attentionRamp[from + 3]
    }
    // Left transparent below the diagonal's ink rather than filled with a
    // background colour: the panel's surface is what shows through, which is
    // how one ramp serves both themes. See `attention-ramp.ts`.
    context.putImageData(new ImageData(pixels, size, size), 0, 0)

    onDrawn(performance.now() - started)
  }, [size, weights, onDrawn])

  function cellAt(clientX: number, clientY: number): AttentionCell | null {
    const canvas = ref.current
    if (!canvas) return null
    const rect = canvas.getBoundingClientRect()
    const key = Math.floor(((clientX - rect.left) / rect.width) * size)
    const query = Math.floor(((clientY - rect.top) / rect.height) * size)
    if (key < 0 || key >= size || query < 0 || query >= size) return null
    return { query, key, weight: weights[query * size + key] }
  }

  return (
    <div className="relative overflow-hidden rounded-(--radius-control) border border-border/45">
      <canvas
        ref={ref}
        role="img"
        aria-label={`Attention from layer ${matrix.layer}, head ${matrix.head}, over ${tokens.length} tokens`}
        // `pixelated`, because the browser's default smoothing would blend
        // neighbouring cells into each other and invent weights between them.
        style={{ imageRendering: "pixelated" }}
        className="block aspect-square w-full cursor-crosshair"
        onPointerMove={(event) =>
          onCellChange(cellAt(event.clientX, event.clientY))
        }
        onPointerLeave={() => onCellChange(null)}
      />
      {cell && (
        <>
          <div
            aria-hidden
            className="pointer-events-none absolute inset-x-0 border-t border-primary/60"
            style={{ top: `${((cell.query + CELL_CENTRE) / size) * PERCENT}%` }}
          />
          <div
            aria-hidden
            className="pointer-events-none absolute inset-y-0 border-l border-primary/60"
            style={{ left: `${((cell.key + CELL_CENTRE) / size) * PERCENT}%` }}
          />
        </>
      )}
    </div>
  )
}
