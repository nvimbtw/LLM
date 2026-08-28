import { datasetColor } from "@/components/training/chart-theme"
import type { Ribbon, RibbonWindow } from "@/lib/training"

/**
 * A sample of the run's actual training windows, one bar each, segmented by the
 * documents inside it and coloured by dataset.
 *
 * The figure `boundary_straddle` is a number for. "63% of windows cross a
 * document boundary" does not distinguish a corpus of books — where a window is
 * one unbroken bar because it is sitting mid-chapter — from a corpus of tweets,
 * where the same number means every window is forty unrelated things. Those two
 * want opposite decisions about the window size and the packing, and only the
 * picture tells them apart.
 *
 * **SVG rather than canvas2d.** The element count is bounded by the engine's own
 * cap (`RIBBON_WINDOWS` × `RIBBON_SEGMENTS`, under a thousand rects), so the
 * usual reason to reach for canvas does not apply and the choice comes down to
 * which produces less code. Three things settle it: the fills are the theme's own
 * `hsl()` values applied as attributes, where canvas would need `getComputedStyle`
 * plus a redraw on every theme switch; a per-segment tooltip is a `<title>` child
 * rather than a hit-test; and this file needs no `'use client'`, no ref and no
 * effect, where a canvas is a client component by construction (R14). Canvas
 * would win if the cap were ever lifted — that is the signal to revisit.
 */
export interface SpanRibbonProps {
  ribbon: Ribbon
  /**
   * The window the segments' token offsets are measured against. It comes from
   * the run's config rather than the frame on purpose: `ctx_window` is not part
   * of the corpus key, so one packed build is legitimately trained at two of
   * them and the engine ships the offsets bare.
   */
  ctxWindow: number
}

// The drawn height of one window, in CSS pixels. The strip is `RIBBON_WINDOWS`
// of these tall — enough that a bar is a bar rather than a hairline, small
// enough that the whole sample is one glance.
const ROW_HEIGHT_PX = 9

// Share of a row the bar fills. The remainder is the gutter that keeps stacked
// windows reading as separate windows rather than as one block of colour.
const BAR_FRACTION = 0.7

// Height of the tail a window's documents overflowed into, as a share of a row.
// Deliberately far under `BAR_FRACTION`: the tail is real tokens that are too
// fine to draw as bars, and at bar height it would read as one long document —
// the exact lie the figure exists to prevent.
const ELIDED_FRACTION = 0.24

// And drawn recessively as well as thin: it stands for tokens nobody measured
// the shape of, so it must not compete with the segments that were.
const ELIDED_OPACITY = 0.35

// Hairline between two documents inside one window, in device pixels. Held at
// device scale by `vector-effect`, because the x axis is stretched to the
// figure's width and the y axis is not.
const BOUNDARY_WIDTH_PX = 1

// The x axis is drawn as a percentage of one window, so the same geometry serves
// any `ctx_window` and the SVG can stretch to whatever width the figure gets.
const FULL_WIDTH = 100

export function SpanRibbon({ ribbon, ctxWindow }: SpanRibbonProps) {
  const windows = ribbon?.windows
  if (!windows?.length || !ctxWindow) return null

  const scale = FULL_WIDTH / ctxWindow
  const barTop = (1 - BAR_FRACTION) / 2
  const elidedTop = (1 - ELIDED_FRACTION) / 2

  return (
    <div className="space-y-2">
      <svg
        viewBox={`0 0 ${FULL_WIDTH} ${windows.length}`}
        preserveAspectRatio="none"
        style={{ height: `${windows.length * ROW_HEIGHT_PX}px` }}
        className="w-full"
        role="img"
        aria-label={`${windows.length} training windows sampled across the corpus, each split into the documents inside it`}
      >
        {windows.map((window, row) => {
          const elidedFrom = drawnUpTo(window) * scale
          return (
            <g key={window.index}>
              <title>{describe(window)}</title>
              {window.segments.map((segment) => (
                <rect
                  key={segment.start}
                  x={segment.start * scale}
                  y={row + barTop}
                  width={segment.length * scale}
                  height={BAR_FRACTION}
                  fill={datasetColor(segment.dataset)}
                />
              ))}
              {window.overflow > 0 && (
                <rect
                  x={elidedFrom}
                  y={row + elidedTop}
                  width={FULL_WIDTH - elidedFrom}
                  height={ELIDED_FRACTION}
                  fill="var(--muted-foreground)"
                  opacity={ELIDED_OPACITY}
                />
              )}
              {/* Only where a document actually ends inside the window. A bar's
                  own left and right edges are the window's, and drawing them
                  like a boundary is what makes one book read as nine
                  documents. */}
              {window.segments
                .filter((s) => s.closes && s.start + s.length < ctxWindow)
                .map((s) => (
                  <line
                    key={`edge-${s.start}`}
                    x1={(s.start + s.length) * scale}
                    x2={(s.start + s.length) * scale}
                    y1={row + barTop}
                    y2={row + barTop + BAR_FRACTION}
                    stroke="var(--background)"
                    strokeWidth={BOUNDARY_WIDTH_PX}
                    vectorEffect="non-scaling-stroke"
                  />
                ))}
            </g>
          )
        })}
      </svg>
      <RibbonKey ribbon={ribbon} />
    </div>
  )
}

/** Where a window's drawn segments stop — the start of the elided tail. */
function drawnUpTo(window: RibbonWindow): number {
  const last = window.segments[window.segments.length - 1]
  return last ? last.start + last.length : 0
}

function describe(window: RibbonWindow): string {
  const drawn = window.segments.length
  const total = drawn + window.overflow
  const documents = `${total.toLocaleString()} document${total === 1 ? "" : "s"}`
  // A window with one document that neither opens nor closes in it is the whole
  // point of the figure, so it gets said in words as well as drawn.
  if (total === 1 && !window.segments[0].opens && !window.segments[0].closes) {
    return `Window ${window.index.toLocaleString()} — inside a single document`
  }
  const elided = window.overflow > 0 ? `, ${drawn.toLocaleString()} drawn` : ""
  return `Window ${window.index.toLocaleString()} — ${documents}${elided}`
}

// Which datasets the sample actually landed on, in the corpus's own order. Not
// every dataset in the build necessarily appears: the sample is 48 windows, and
// a dataset contributing a thousandth of the tokens will often miss them all.
function RibbonKey({ ribbon }: { ribbon: Ribbon }) {
  const seen = new Set<number>()
  for (const window of ribbon.windows) {
    for (const segment of window.segments) seen.add(segment.dataset)
  }
  const shown = [...seen].sort((a, b) => a - b)

  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
      {shown.map((dataset) => (
        <li key={dataset} className="flex items-center gap-1.5">
          <span
            aria-hidden
            className="inline-block size-2 rounded-(--radius-marker)"
            style={{ backgroundColor: datasetColor(dataset) }}
          />
          {ribbon.datasets[dataset] ?? `Dataset ${dataset}`}
        </li>
      ))}
    </ul>
  )
}
