"use client"
import { useId, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import type { GraphFragment } from "@/lib/architecture/contracts.generated"
import type { GraphLayout } from "@/lib/architecture-graph"
export interface GraphCanvasProps {
  fragment: GraphFragment
  layout: GraphLayout
  selected: string | null
  onSelect: (id: string) => void
}
/** Displays a bounded graph page with pointer/keyboard panning, zoom and fit; semantic inspection lives in the equivalent list. */
export function GraphCanvas({
  fragment,
  layout,
  selected,
  onSelect,
}: GraphCanvasProps) {
  const arrowId = useId()
  const [viewport, setViewport] = useState({ zoom: 1, x: 0, y: 0 })
  const drag = useRef<{
    x: number
    y: number
    moved: boolean
    node: string | null
  } | null>(null)
  return (
    <>
      <div className="flex flex-wrap gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() =>
            setViewport((current) => ({
              ...current,
              zoom: Math.max(0.25, current.zoom / 1.25),
            }))
          }
        >
          Zoom out
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() =>
            setViewport((current) => ({
              ...current,
              zoom: Math.min(128, current.zoom * 1.25),
            }))
          }
        >
          Zoom in
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setViewport({ zoom: 1, x: 0, y: 0 })}
        >
          Fit graph
        </Button>
        <span className="self-center text-sm text-muted-foreground">
          {Math.round(viewport.zoom * 100)}%
        </span>
      </div>
      <div
        tabIndex={0}
        role="region"
        aria-label="Graph canvas. Arrow keys pan; plus and minus zoom; zero fits."
        className="h-96 touch-none overflow-hidden rounded-(--radius-sheet) border border-border focus-visible:outline-2 focus-visible:outline-ring"
        onPointerDown={(event) => {
          if (event.button === 0) {
            drag.current = {
              x: event.clientX,
              y: event.clientY,
              moved: false,
              node:
                event.target instanceof Element
                  ? (event.target
                      .closest("[data-node-id]")
                      ?.getAttribute("data-node-id") ?? null)
                  : null,
            }
            event.currentTarget.setPointerCapture(event.pointerId)
          }
        }}
        onPointerMove={(event) => {
          if (!drag.current) return
          const scale = Math.min(
            event.currentTarget.clientWidth / layout.width,
            event.currentTarget.clientHeight / layout.height
          )
          const ratio = 1 / Math.max(Number.EPSILON, scale)
          const delta = {
            x: (event.clientX - drag.current.x) * ratio,
            y: (event.clientY - drag.current.y) * ratio,
          }
          drag.current = {
            ...drag.current,
            x: event.clientX,
            y: event.clientY,
            moved:
              drag.current.moved || Math.abs(delta.x) + Math.abs(delta.y) > 2,
          }
          setViewport((current) => ({
            ...current,
            x: current.x + delta.x,
            y: current.y + delta.y,
          }))
        }}
        onPointerUp={() => {
          if (drag.current && !drag.current.moved && drag.current.node)
            onSelect(drag.current.node)
          drag.current = null
        }}
        onPointerCancel={() => {
          drag.current = null
        }}
        onKeyDown={(event) => {
          if (event.ctrlKey || event.metaKey || event.altKey) return
          if (
            ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(
              event.key
            )
          ) {
            event.preventDefault()
            setViewport((current) => ({
              ...current,
              x:
                current.x +
                (event.key === "ArrowLeft"
                  ? 40
                  : event.key === "ArrowRight"
                    ? -40
                    : 0),
              y:
                current.y +
                (event.key === "ArrowUp"
                  ? 40
                  : event.key === "ArrowDown"
                    ? -40
                    : 0),
            }))
          }
          if (event.key === "+" || event.key === "=")
            setViewport((current) => ({
              ...current,
              zoom: Math.min(128, current.zoom * 1.25),
            }))
          if (event.key === "-")
            setViewport((current) => ({
              ...current,
              zoom: Math.max(0.25, current.zoom / 1.25),
            }))
          if (event.key === "0") setViewport({ zoom: 1, x: 0, y: 0 })
        }}
      >
        <svg
          aria-hidden="true"
          width="100%"
          height="100%"
          viewBox={`0 0 ${layout.width} ${layout.height}`}
        >
          <defs>
            <marker
              id={arrowId}
              markerWidth="8"
              markerHeight="8"
              refX="7"
              refY="4"
              orient="auto"
            >
              <path d="M 0 0 L 8 4 L 0 8" fill="none" stroke="currentColor" />
            </marker>
          </defs>
          <g
            transform={`translate(${viewport.x} ${viewport.y}) scale(${viewport.zoom})`}
          >
            {fragment.edges.map((edge, index) => {
              const from = layout.nodes.find(
                (item) => item.node.id === edge.source
              )
              const to = layout.nodes.find(
                (item) => item.node.id === edge.target
              )
              return from && to ? (
                <path
                  key={index}
                  d={`M ${from.x + 224} ${from.y + 40} L ${to.x} ${to.y + 40}`}
                  fill="none"
                  markerEnd={`url(#${arrowId})`}
                  stroke="currentColor"
                  className="text-border"
                />
              ) : null
            })}
            {layout.nodes.map(({ node, x, y }) => (
              <g
                key={node.id}
                transform={`translate(${x} ${y})`}
                data-node-id={node.id}
              >
                <rect
                  width={224}
                  height={80}
                  rx={4}
                  className={
                    selected === node.id
                      ? "fill-accent stroke-ring"
                      : "fill-card stroke-border"
                  }
                />
                <text x={12} y={24} className="fill-foreground text-sm">
                  {node.label.slice(0, 27)}
                </text>
                <text x={12} y={44} className="fill-muted-foreground text-xs">
                  {node.kind}
                  {node.iterations == null
                    ? ""
                    : ` · ${node.iterations} iterations`}
                </text>
                <text x={12} y={64} className="fill-muted-foreground text-xs">
                  {node.parameter_owner_ids?.length
                    ? node.parameter_owner_ids.join(", ").slice(0, 28)
                    : "No parameter owners"}
                </text>
              </g>
            ))}
          </g>
        </svg>
      </div>
    </>
  )
}
