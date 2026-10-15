import { memo, useId, useMemo } from "react"

import {
  architectureLayout,
  diagramPath,
  operationProgress,
  phaseProgress,
  smooth,
  stageEmphasis,
  pointOnRoute,
  port,
  promptTokens,
  unit,
  type ArchitectureFrame,
  type ArchitectureStep,
  type DiagramBox,
  type DiagramPoint,
  type Operation,
} from "@/lib/architecture-flow"
import type { ArchitectureFlowProps } from "./architecture-flow"
import { ArchitectureOperation } from "./architecture-operation"

export interface ArchitectureDiagramProps extends ArchitectureFlowProps {
  vertical: boolean
  frame: ArchitectureFrame | null
}

type Layout = ReturnType<typeof architectureLayout>
interface CircuitProps extends ArchitectureDiagramProps {
  layout: Layout
  arrow: string
}
interface EdgeProps {
  points: DiagramPoint[]
  arrow: string
  frame: ArchitectureFrame | null
  step: ArchitectureStep
  start?: number
  end?: number
  dashed?: boolean
  tokenWidth?: number
}

const FORWARD: Operation[] = [
  "embedding",
  "attention",
  "feedforward",
  "norm",
  "output",
]
const PHASE_LABELS: Record<ArchitectureStep, string> = {
  input: "Read the sequence",
  embedding: "Tokens become vectors",
  attention: "Mix information across positions",
  feedforward: "Transform each position",
  norm: "Normalize the residual stream",
  output: "Project to token scores",
  sample: "Select the next token",
  append: "Append to the prompt",
  loss: "Compare predictions with supervision",
  backward: "Compute weight gradients",
  optimizer: "Calculate weight updates",
  update: "Apply updates across the model",
  batch: "Read the next training batch",
}

/** Renders one responsive schematic from shared ports and phase data. Geometry is bounded independently of layer/head/context counts. */
export function ArchitectureDiagram({
  cfg,
  view,
  frame,
  vertical,
}: ArchitectureDiagramProps) {
  const id = useId().replaceAll(":", "")
  const layout = useMemo(() => architectureLayout(vertical), [vertical])
  const { nodes, stack, prompt } = layout
  const arrow = `url(#${id}-arrow)`
  const current = frame ?? { step: "input", progress: 0, cycle: 0 }
  const sequence = promptTokens(current, view)
  const input = port(nodes.embedding, "top")
  const inputPoints = vertical
    ? [
        { x: prompt.x + 44, y: prompt.y + 8 },
        { x: prompt.x + 44, y: 80 },
        { x: input.x, y: 80 },
        input,
      ]
    : [{ x: input.x, y: prompt.y + 8 }, input]
  const height = vertical
    ? view === "generation"
      ? 974
      : 1260
    : view === "generation"
      ? 456
      : 528
  const operations: Operation[] = [
    ...FORWARD,
    ...(view === "generation"
      ? ["sample" as const]
      : ["loss" as const, "backward" as const, "optimizer" as const]),
  ]
  const labels = operationLabels(cfg)
  const circuit = { cfg, view, frame, vertical, layout, arrow }
  return (
    <svg
      viewBox={`0 0 ${layout.width} ${height}`}
      className={vertical ? "mx-auto block w-full max-w-sm" : "block w-full"}
      role="img"
      aria-labelledby={`${id}-title ${id}-description`}
      data-diagram-layout={vertical ? "vertical" : "horizontal"}
    >
      <title id={`${id}-title`}>
        {view === "generation"
          ? "Model generation flow"
          : "Model training flow"}
      </title>
      <desc id={`${id}-description`}>
        Illustrative flow, not live model activity. Up to{" "}
        {metric(cfg.ctx_window)} context tokens become {metric(cfg.dimensions)}{" "}
        dimensional embeddings. {metric(cfg.layers)} transformer layers each
        apply pre-normalized causal attention with {metric(cfg.heads)} heads and
        RoPE, then a {metric(cfg.dimensions * cfg.ff_multiplier)} wide GELU
        feed-forward network. Each sublayer adds its residual input. Final
        normalization and a projection tied to the embedding weights produce
        token scores.
        {view === "generation"
          ? " Sampling selects one token and appends it to the prompt before the next forward pass. The visible strip shows a bounded excerpt of the growing sequence; caching is omitted."
          : ` ${labels.loss.detail} uses ${labels.loss.note?.toLowerCase()}. Backpropagation computes gradients; the optimizer updates weights throughout the model before the next batch.`}
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
        <clipPath id={`${id}-prompt`}>
          <rect x={prompt.x} y={prompt.y - 8} width="278" height="16" />
        </clipPath>
      </defs>
      <TransformerStack
        box={stack}
        layers={cfg.layers}
        emphasis={stageEmphasis(frame, "update")}
      />
      <Edge points={inputPoints} arrow={arrow} frame={frame} step="embedding" />
      <ForwardConnections {...circuit} />
      {view === "generation" ? (
        <GenerationReturn {...circuit} next={sequence.next} />
      ) : (
        <TrainingReturn {...circuit} />
      )}
      {operations.map((operation) => (
        <OperationBlock
          key={operation}
          box={nodes[operation]}
          operation={operation}
          {...labels[operation]}
          progress={operationProgress(frame, operation, view)}
          emphasis={Math.max(
            stageEmphasis(frame, operation),
            FORWARD.includes(operation) ? stageEmphasis(frame, "update") : 0
          )}
        />
      ))}
      <text x={prompt.x} y="17" className="fill-foreground text-xs">
        {view === "generation" ? "Prompt" : "Training batch"}
      </text>
      <text
        x={prompt.x + 278}
        y="17"
        textAnchor="end"
        className="fill-muted-foreground text-2xs"
      >
        {metric(cfg.ctx_window)} context
      </text>
      <g clipPath={`url(#${id}-prompt)`}>
        {sequence.tokens.map((token) => (
          <rect
            key={token.index}
            x={prompt.x + token.x}
            y={prompt.y - 7}
            width={token.width}
            height="14"
            rx="1"
            style={{
              fill: `color-mix(in oklab, var(--primary) ${stageEmphasis(frame, "input") * 100}%, var(--muted-foreground))`,
            }}
            opacity={
              frame?.step === "batch"
                ? smooth(Math.abs(frame.progress * 2 - 1)) * 0.4
                : 0.4 + stageEmphasis(frame, "input") * 0.2
            }
          />
        ))}
        {view === "generation" &&
          frame?.step === "append" &&
          frame.progress >= 0.8 && (
            <rect
              x={prompt.x + sequence.next.x}
              y={prompt.y - 7}
              width={sequence.next.width}
              height="14"
              rx="1"
              style={{
                fill: `color-mix(in oklab, var(--primary) ${(1 - phaseProgress(frame.progress, 0.8, 1)) * 100}%, var(--muted-foreground))`,
              }}
              opacity={1 - 0.6 * phaseProgress(frame.progress, 0.8, 1)}
            />
          )}
      </g>
      {sequence.clipped && (
        <text
          x={prompt.x - 12}
          y={prompt.y + 4}
          className="fill-muted-foreground text-xs"
        >
          …
        </text>
      )}
      <text
        x={vertical ? 36 : 24}
        y={height - 12}
        className="fill-muted-foreground text-xs"
        data-flow-phase={frame?.step ?? "static"}
      >
        {frame
          ? PHASE_LABELS[frame.step]
          : view === "generation"
            ? "Sample → append → repeat"
            : "Forward → loss → gradients → update"}
      </text>
    </svg>
  )
}

/** Draws a shallow flat stack, retaining the exact layer count even when only a few representative sheets are visible. */
const TransformerStack = memo(function TransformerStack({
  box,
  layers,
  emphasis,
}: {
  box: DiagramBox
  layers: number
  emphasis: number
}) {
  const sheets =
    Number.isSafeInteger(layers) && layers > 0 ? Math.min(layers, 4) : 1
  return (
    <g data-transformer-stack="">
      {Array.from({ length: sheets }, (_, index) => sheets - index - 1).map(
        (index) => (
          <rect
            key={index}
            x={box.x + index * 6}
            y={box.y - index * 6}
            width={box.width}
            height={box.height}
            rx="2"
            className="fill-background"
            style={{
              stroke: `color-mix(in oklab, var(--primary) ${emphasis * 100}%, var(--border))`,
            }}
          />
        )
      )}
      <text x={box.x + 16} y={box.y + 25} className="fill-foreground text-xs">
        Transformer
      </text>
      <text
        x={box.x + box.width - 16}
        y={box.y + 25}
        textAnchor="end"
        className="fill-foreground text-xs font-medium tabular-nums"
      >
        × {metric(layers)} layers
      </text>
    </g>
  )
})

/** Uses the same route for stationary arrows and traveling vector/token marks. Processing starts only after arrival, and return paths have an explicit endpoint. */
function Edge({
  points,
  arrow,
  frame,
  step,
  start = 0,
  end = 0.3,
  dashed = false,
  tokenWidth,
}: EdgeProps) {
  const active =
    frame?.step === step && frame.progress >= start && frame.progress < end
  const progress = frame ? unit((frame.progress - start) / (end - start)) : 0
  const point = pointOnRoute(points, smooth(progress))
  return (
    <g>
      <path
        d={diagramPath(points)}
        fill="none"
        className={dashed ? "stroke-border" : "stroke-muted-foreground/60"}
        strokeDasharray={dashed ? "3 5" : undefined}
        markerEnd={arrow}
      />
      {active && (
        <rect
          x={point.x - (tokenWidth ?? 6) / 2}
          y={point.y - (tokenWidth ? 7 : 3)}
          width={tokenWidth ?? 6}
          height={tokenWidth ? 14 : 6}
          rx="1"
          className="fill-primary"
          opacity={
            smooth(progress / 0.12) *
            (tokenWidth ? 1 : smooth((1 - progress) / 0.15))
          }
        />
      )}
    </g>
  )
}

/** Connects the shared forward ports and draws the two residual additions outside the operation boxes. */
function ForwardConnections({ layout, vertical, arrow, frame }: CircuitProps) {
  const { nodes } = layout
  return (
    <>
      {FORWARD.slice(1).map((operation, index) => {
        const from = port(nodes[FORWARD[index]], vertical ? "bottom" : "right")
        const to = port(nodes[operation], vertical ? "top" : "left")
        return (
          <Edge
            key={operation}
            points={[from, to]}
            arrow={arrow}
            frame={frame}
            step={operation}
          />
        )
      })}
      {(["attention", "feedforward"] as const).map((operation) => {
        const box = nodes[operation]
        const input = port(box, vertical ? "top" : "left")
        const output = port(box, vertical ? "bottom" : "right")
        const add = vertical
          ? { x: output.x, y: output.y + 22 }
          : { x: output.x + 24, y: output.y }
        const points = vertical
          ? [
              { x: input.x, y: input.y - 12 },
              { x: 316, y: input.y - 12 },
              { x: 316, y: add.y },
              { x: add.x + 8, y: add.y },
            ]
          : [
              { x: input.x - 12, y: input.y },
              { x: input.x - 12, y: 130 },
              { x: add.x, y: 130 },
              { x: add.x, y: add.y - 8 },
            ]
        return (
          <g key={operation}>
            <Edge
              points={points}
              arrow={arrow}
              frame={frame}
              step={operation}
              start={0.72}
              end={1}
            />
            <circle
              cx={add.x}
              cy={add.y}
              r="8"
              className="fill-background stroke-border"
            />
            <path
              d={`M${add.x - 3} ${add.y}h6 M${add.x} ${add.y - 3}v6`}
              className="stroke-muted-foreground"
            />
          </g>
        )
      })}
    </>
  )
}

/** Routes one sampled token back to its precise append slot; prompt overflow scrolls before the returning token arrives. */
function GenerationReturn({
  layout,
  vertical,
  arrow,
  frame,
  next,
}: CircuitProps & { next: { x: number; width: number } }) {
  const { nodes, prompt } = layout
  const start = port(nodes.sample, "right")
  const side = vertical ? 364 : 896
  return (
    <>
      <Edge
        points={[port(nodes.output, "bottom"), port(nodes.sample, "top")]}
        arrow={arrow}
        frame={frame}
        step="sample"
      />
      <Edge
        points={[
          start,
          { x: side, y: start.y },
          { x: side, y: prompt.y },
          { x: prompt.x + next.x + next.width / 2, y: prompt.y },
        ]}
        arrow={arrow}
        frame={frame}
        step="append"
        end={0.8}
        tokenWidth={next.width}
      />
      {!vertical && (
        <text x="594" y="28" className="fill-muted-foreground text-2xs">
          Append selected token
        </text>
      )}
    </>
  )
}

/** Separates target supervision from predicted scores and distributes optimizer updates to every trainable stage before a new batch. */
function TrainingReturn({ cfg, layout, vertical, arrow, frame }: CircuitProps) {
  const { nodes, stack } = layout
  const output = port(nodes.output, vertical ? "bottom" : "right")
  const loss = port(nodes.loss, vertical ? "top" : "right")
  const prediction = vertical
    ? [output, loss]
    : [output, { x: 896, y: output.y }, { x: 896, y: loss.y }, loss]
  const optimizer = port(nodes.optimizer, vertical ? "right" : "top")
  const updateTargets = [nodes.embedding, stack, nodes.norm, nodes.output]
  const target = port(nodes.loss, vertical ? "left" : "bottom")
  return (
    <>
      <Edge points={prediction} arrow={arrow} frame={frame} step="loss" />
      {(["backward", "optimizer"] as const).map((operation, index) => (
        <Edge
          key={operation}
          points={[
            port(
              nodes[index ? "backward" : "loss"],
              vertical ? "bottom" : "left"
            ),
            port(nodes[operation], vertical ? "top" : "right"),
          ]}
          arrow={arrow}
          frame={frame}
          step={operation}
        />
      ))}
      {updateTargets.map((box, index) => {
        const end = port(box, vertical ? "right" : "bottom")
        const points = vertical
          ? [optimizer, { x: 364, y: optimizer.y }, { x: 364, y: end.y }, end]
          : [optimizer, { x: optimizer.x, y: 294 }, { x: end.x, y: 294 }, end]
        return (
          <Edge
            key={index}
            points={points}
            arrow={arrow}
            frame={frame}
            step="update"
            end={0.8}
            dashed
          />
        )
      })}
      <text
        x={vertical ? 336 : 24}
        y={vertical ? 1090 : 316}
        textAnchor={vertical ? "end" : "start"}
        className="fill-muted-foreground text-2xs"
      >
        Weight updates
      </text>
      <Edge
        points={
          vertical
            ? [{ x: 40, y: target.y }, target]
            : [{ x: target.x, y: target.y + 38 }, target]
        }
        arrow={arrow}
        frame={frame}
        step="loss"
      />
      <text
        x={vertical ? 36 : target.x}
        y={vertical ? target.y - 18 : target.y + 60}
        textAnchor={vertical ? "start" : "middle"}
        className="fill-muted-foreground text-2xs"
      >
        {cfg.mode === "rl"
          ? "Rewards"
          : cfg.mode === "kto"
            ? "Feedback"
            : "Targets"}
      </text>
    </>
  )
}

/** Renders labels and an operation-specific miniature while preserving neutral page surfaces. */
const OperationBlock = memo(function OperationBlock({
  box,
  operation,
  label,
  detail,
  note,
  progress,
  emphasis,
}: {
  box: DiagramBox
  operation: Operation
  label: string
  detail: string
  note?: string
  progress: number
  emphasis: number
}) {
  return (
    <g
      transform={`translate(${box.x} ${box.y})`}
      data-flow-node={operation}
      data-active={emphasis > 0}
      data-operation-progress={progress}
    >
      <rect
        width={box.width}
        height={box.height}
        rx="2"
        className="fill-background stroke-border"
      />
      <rect
        width={box.width}
        height={box.height}
        rx="2"
        className="fill-primary/5 stroke-primary"
        opacity={emphasis}
      />
      <text
        x={box.width / 2}
        y="20"
        textAnchor="middle"
        className="fill-foreground text-xs"
      >
        {label}
      </text>
      <text
        x={box.width / 2}
        y="37"
        textAnchor="middle"
        className="fill-muted-foreground text-2xs"
      >
        {detail}
      </text>
      {note && (
        <text
          x={box.width / 2}
          y="51"
          textAnchor="middle"
          className="fill-muted-foreground text-2xs"
        >
          {note}
        </text>
      )}
      <g transform={`translate(${(box.width - 80) / 2} 62)`}>
        <ArchitectureOperation
          operation={operation}
          progress={progress}
          emphasis={emphasis}
        />
      </g>
    </g>
  )
})

/** Describes configured shape and the supervision used by the current training family; generation is independent of that family. */
function operationLabels(
  cfg: ArchitectureFlowProps["cfg"]
): Record<Operation, { label: string; detail: string; note?: string }> {
  const loss =
    cfg.mode === "rl"
      ? "Policy objective"
      : cfg.mode === "kto"
        ? "Preference loss"
        : cfg.mode === "sft"
          ? "Masked CE"
          : "Cross entropy"
  const targets =
    cfg.mode === "rl"
      ? "Rewards"
      : cfg.mode === "kto"
        ? "Feedback"
        : cfg.mode === "sft"
          ? "Assistant targets"
          : "Next-token targets"
  return {
    embedding: {
      label: "Embedding",
      detail: `${metric(cfg.dimensions)} channels`,
    },
    attention: {
      label: "Attention",
      detail: `${metric(cfg.heads)} heads`,
      note: "Pre-norm · RoPE",
    },
    feedforward: {
      label: "Feed-forward",
      detail: `${metric(cfg.dimensions * cfg.ff_multiplier)} hidden`,
      note: "Pre-norm · GELU",
    },
    norm: { label: "Final norm", detail: `${metric(cfg.dimensions)} channels` },
    output: {
      label: "Output",
      detail: "Token scores",
      note: "Tied projection",
    },
    sample: { label: "Sampling", detail: "One next token" },
    loss: { label: "Loss", detail: loss, note: targets },
    backward: { label: "Backward", detail: "Weight gradients" },
    optimizer: { label: "Optimizer", detail: "Update weights" },
  }
}

/** Keeps invalid draft dimensions unavailable rather than presenting them as real measurements. */
function metric(value: number): string {
  return Number.isSafeInteger(value) && value > 0 ? value.toLocaleString() : "—"
}
