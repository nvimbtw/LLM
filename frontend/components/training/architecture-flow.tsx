"use client"

import { useId, useState } from "react"
import { Pause, Play } from "lucide-react"

import { Button } from "@/components/ui/button"
import { usePrefersReducedMotion } from "@/hooks/use-reduced-motion"
import type { TrainConfig } from "@/lib/training"

export interface ArchitectureFlowProps {
  cfg: Pick<
    TrainConfig,
    "dimensions" | "ctx_window" | "heads" | "layers" | "ff_multiplier"
  > & {
    mode: TrainConfig["mode"] | "rl"
  }
}

interface FlowDiagramProps extends ArchitectureFlowProps {
  vertical: boolean
  animated: boolean
}

interface FlowNodeProps {
  x: number
  y: number
  width: number
  label: string
  detail: string
  note?: string
  activationDelaySeconds?: number
}

/** Illustrates the configured forward pass with bounded SVG geometry. Motion is decorative, can be paused, and follows the browser's reduced-motion preference. */
export function ArchitectureFlow({ cfg }: ArchitectureFlowProps) {
  const reducedMotion = usePrefersReducedMotion()
  const [paused, setPaused] = useState(false)
  return (
    <figure className="mt-6" aria-label="Model architecture">
      <figcaption className="mb-3 flex items-center justify-between gap-3 text-xs text-muted-foreground">
        <span>Training loop</span>
        {!reducedMotion && (
          <Button
            type="button"
            variant="text"
            size="xs"
            className="text-muted-foreground"
            aria-label={
              paused
                ? "Play architecture animation"
                : "Pause architecture animation"
            }
            onClick={() => setPaused(!paused)}
          >
            {paused ? <Play /> : <Pause />}
            {paused ? "Play" : "Pause"}
          </Button>
        )}
      </figcaption>
      <FlowDiagram
        cfg={cfg}
        vertical={false}
        animated={!paused && !reducedMotion}
      />
      <FlowDiagram cfg={cfg} vertical animated={!paused && !reducedMotion} />
    </figure>
  )
}

/** Draws one representative pre-norm block with its repeat count, residual branches and tied output, laid out for the available screen width. */
function FlowDiagram({ cfg, vertical, animated }: FlowDiagramProps) {
  const id = useId().replaceAll(":", "")
  const dFf = cfg.dimensions * cfg.ff_multiplier
  const lossLabel =
    cfg.mode === "rl"
      ? "Policy objective"
      : cfg.mode === "kto"
        ? "Preference"
        : cfg.mode === "sft"
          ? "Masked CE"
          : "Cross entropy"
  const targetLabel =
    cfg.mode === "rl"
      ? "Rewards"
      : cfg.mode === "kto"
        ? "Feedback"
        : cfg.mode === "sft"
          ? "Assistant tokens"
          : "Next token"
  const nodes: FlowNodeProps[] = vertical
    ? [
        {
          x: 100,
          y: 0,
          width: 140,
          label: "Tokens",
          detail: `${metric(cfg.ctx_window)} context`,
        },
        {
          x: 100,
          y: 84,
          width: 140,
          label: "Embedding",
          detail: `${metric(cfg.dimensions)} channels`,
        },
        {
          x: 100,
          y: 210,
          width: 140,
          label: "Attention",
          detail: `${metric(cfg.heads)} heads`,
          note: "Pre-norm · RoPE",
        },
        {
          x: 100,
          y: 320,
          width: 140,
          label: "Feed-forward",
          detail: `${metric(dFf)} hidden`,
          note: "Pre-norm · GELU",
        },
        {
          x: 100,
          y: 468,
          width: 140,
          label: "Final norm",
          detail: `${metric(cfg.dimensions)} channels`,
        },
        { x: 100, y: 552, width: 140, label: "Output", detail: "Token scores" },
        { x: 0, y: 656, width: 76, label: "Targets", detail: targetLabel },
        { x: 100, y: 656, width: 140, label: "Loss", detail: lossLabel },
        { x: 100, y: 756, width: 140, label: "Backward", detail: "Gradients" },
        {
          x: 100,
          y: 856,
          width: 140,
          label: "Gradients",
          detail: "Weight derivatives",
        },
        {
          x: 100,
          y: 956,
          width: 140,
          label: "Optimizer",
          detail: "Update weights",
        },
      ]
    : [
        {
          x: 0,
          y: 98,
          width: 96,
          label: "Tokens",
          detail: metric(cfg.ctx_window),
        },
        {
          x: 144,
          y: 98,
          width: 96,
          label: "Embedding",
          detail: `${metric(cfg.dimensions)} channels`,
        },
        {
          x: 288,
          y: 98,
          width: 96,
          label: "Attention",
          detail: `${metric(cfg.heads)} heads`,
          note: "Pre-norm · RoPE",
        },
        {
          x: 432,
          y: 98,
          width: 96,
          label: "Feed-forward",
          detail: `${metric(dFf)} hidden`,
          note: "Pre-norm · GELU",
        },
        {
          x: 576,
          y: 98,
          width: 96,
          label: "Final norm",
          detail: `${metric(cfg.dimensions)}d`,
        },
        { x: 720, y: 98, width: 96, label: "Output", detail: "Scores" },
        { x: 720, y: 266, width: 96, label: "Targets", detail: targetLabel },
        { x: 576, y: 266, width: 96, label: "Loss", detail: lossLabel },
        { x: 432, y: 266, width: 96, label: "Backward", detail: "Chain rule" },
        {
          x: 288,
          y: 266,
          width: 96,
          label: "Gradients",
          detail: "Weight derivatives",
        },
        {
          x: 144,
          y: 266,
          width: 96,
          label: "Optimizer",
          detail: "Update weights",
        },
      ]
  const connections = vertical
    ? "M170 56V84 M170 140V210 M170 266V274 M170 298V320 M170 376V384 M170 408V468 M170 524V552"
    : "M96 126H144 M240 126H288 M384 126H398 M418 126H432 M528 126H542 M562 126H576 M672 126H720"
  const residuals = vertical
    ? "M170 198H290V286H182 M170 308H290V396H182"
    : "M276 126V82H408V116 M418 126V82H552V116"
  const additions = vertical
    ? [
        [170, 286],
        [170, 396],
      ]
    : [
        [408, 126],
        [552, 126],
      ]

  return (
    <svg
      viewBox={vertical ? "-4 -4 348 1040" : "-4 0 852 354"}
      className={vertical ? "block w-full sm:hidden" : "hidden w-full sm:block"}
      role="img"
      aria-labelledby={`${id}-title ${id}-description`}
    >
      <title id={`${id}-title`}>Configured model training loop</title>
      <desc id={`${id}-description`}>
        Tokens in a {metric(cfg.ctx_window)} token context become{" "}
        {metric(cfg.dimensions)} dimensional embeddings. A pre-norm attention
        and feed-forward block repeats {metric(cfg.layers)} times, with{" "}
        {metric(cfg.heads)} attention heads and {metric(dFf)} feed-forward
        width. Both sublayers add to the residual stream. Final normalization
        precedes token scores, projected using the embedding weights. Target
        tokens and scores produce {lossLabel.toLowerCase()} loss.
        Backpropagation computes gradients; the optimizer updates model weights.
      </desc>
      <defs>
        <marker
          id={`${id}-arrow`}
          viewBox="0 0 6 6"
          refX="6"
          refY="3"
          markerWidth="5"
          markerHeight="5"
          orient="auto-start-reverse"
        >
          <path d="M0 0L6 3L0 6" className="fill-muted-foreground" />
        </marker>
      </defs>
      <rect
        x={vertical ? 28 : 264}
        y={vertical ? 178 : 42}
        width={vertical ? 284 : 308}
        height={vertical ? 250 : 142}
        rx="3"
        fill="none"
        className="stroke-border"
      />
      <text
        x={vertical ? 44 : 280}
        y={vertical ? 200 : 64}
        className="fill-muted-foreground text-xs"
      >
        Transformer block × {metric(cfg.layers)}
      </text>
      <path
        d={connections}
        fill="none"
        className="stroke-muted-foreground"
        markerEnd={`url(#${id}-arrow)`}
      />
      <path
        d={residuals}
        fill="none"
        className="stroke-muted-foreground/60"
        markerEnd={`url(#${id}-arrow)`}
      />
      <path
        d={vertical ? "M240 112H328V580H240" : "M192 154V194H768V154"}
        fill="none"
        strokeDasharray="3 5"
        className="stroke-border"
      />
      <path
        d={
          vertical
            ? "M170 608V656 M76 684H100 M170 712V756 M170 812V856 M170 912V956 M240 984H320V112H240 M320 304H312 M320 496H240"
            : "M816 126H840V246H624V266 M720 294H672 M576 294H528 M432 294H384 M288 294H240 M192 266V218H120V12H192V98 M120 12H336V42 M336 12H624V98"
        }
        fill="none"
        className="stroke-muted-foreground"
        markerEnd={`url(#${id}-arrow)`}
      />
      {animated && (
        <circle r="2.5" className="fill-primary">
          <animateMotion
            id={`${id}-signal`}
            path={
              vertical
                ? "M170 0V984H320V112H170"
                : "M0 126H840V246H624V294H192V218H120V12H192V126"
            }
            dur="12s"
            repeatCount="indefinite"
          />
        </circle>
      )}
      {nodes.map((node) => (
        <FlowNode
          key={node.label}
          {...node}
          activationDelaySeconds={
            animated ? activationTime(node, vertical) : undefined
          }
        />
      ))}
      {additions.map(([x, y]) => (
        <g key={x + ":" + y}>
          <circle
            cx={x}
            cy={y}
            r={vertical ? 12 : 10}
            className="fill-background stroke-border"
          />
          <text
            x={x}
            y={y + 4}
            textAnchor="middle"
            className="fill-foreground text-sm"
          >
            +
          </text>
        </g>
      ))}
    </svg>
  )
}

/** Renders one operation and its current dimensions in a quiet schematic block. */
function FlowNode({
  x,
  y,
  width,
  label,
  detail,
  note,
  activationDelaySeconds,
}: FlowNodeProps) {
  return (
    <g data-flow-node={label}>
      <rect
        x={x}
        y={y}
        width={width}
        height="56"
        rx="3"
        className="fill-background stroke-border"
      />
      {activationDelaySeconds !== undefined && (
        <rect
          x={x}
          y={y}
          width={width}
          height="56"
          rx="3"
          opacity="0"
          className="fill-primary stroke-primary"
        >
          <animate
            attributeName="opacity"
            values="0;0;0.16;0.05;0;0"
            keyTimes={reactionPhases(activationDelaySeconds)}
            dur="12s"
            repeatCount="indefinite"
          />
        </rect>
      )}
      <text
        x={x + width / 2}
        y={y + (note ? 17 : 23)}
        textAnchor="middle"
        className="fill-foreground text-xs"
      >
        {label}
      </text>
      <text
        x={x + width / 2}
        y={y + (note ? 33 : 41)}
        textAnchor="middle"
        className="fill-muted-foreground text-2xs"
      >
        {detail}
      </text>
      {note && (
        <text
          x={x + width / 2}
          y={y + 47}
          textAnchor="middle"
          className="fill-muted-foreground text-2xs"
        >
          {note}
        </text>
      )}
    </g>
  )
}

/** Places a 1.4-second block reaction on the same twelve-second SVG clock as the moving signal, including every repeated cycle. */
function reactionPhases(delaySeconds: number): string {
  return [
    0,
    delaySeconds / 12,
    (delaySeconds + 0.35) / 12,
    (delaySeconds + 0.98) / 12,
    (delaySeconds + 1.4) / 12,
    1,
  ].join(";")
}

/** Times each block's reaction to the signal's arrival on the fixed twelve-second training-loop path. Targets react alongside loss because they feed its second input. */
function activationTime(node: FlowNodeProps, vertical: boolean): number {
  if (vertical) return Math.max(0, ((node.y + 28) / 2156) * 12 - 0.2)
  const center = node.x + node.width / 2
  const distance =
    node.y < 200
      ? center
      : node.label === "Targets"
        ? 1224
        : 1224 + 624 - center
  return Math.max(0, (distance / 2196) * 12 - 0.2)
}

/** Formats meaningful positive dimensions without displaying invalid draft values as a real measurement. */
function metric(value: number | null): string {
  return value !== null && Number.isFinite(value) && value > 0
    ? value.toLocaleString()
    : "—"
}
