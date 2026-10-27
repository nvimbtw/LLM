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

/** Draws one bounded matrix and exposes pointer-selected token pairs without changing the underlying probabilities. */
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
    context.putImageData(new ImageData(pixels, size, size), 0, 0)

    onDrawn(performance.now() - started)
  }, [size, weights, onDrawn])

  /** Maps a pointer position to a bounded actual matrix cell, or returns no selection outside the canvas. */
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
        aria-label={`Attention from ${matrix.pointId ? "selected model view" : `layer ${matrix.layer}`}, head ${matrix.head}, over ${tokens.length} tokens`}
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
