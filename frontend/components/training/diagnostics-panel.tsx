"use client"

import * as React from "react"
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts"

import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  ACCENT_COLOR,
  axisProps,
  EMB_COLOR,
  gridProps,
  headColor,
  layerColor,
  MUTED_SERIES,
  yAxisProps,
} from "@/components/training/chart-theme"
import { Figure, FigurePlaceholder } from "@/components/training/stat-run"
import type { DiagPoint } from "@/lib/training-socket"

// A positive-only reading for a log axis: non-finite or ≤0 values become null so
// the point is dropped rather than breaking the log scale (recharts can't place 0
// on a log axis). Used for the grad-norm and update-ratio figures.
const pos = (v: number): number | null =>
  Number.isFinite(v) && v > 0 ? v : null

// A finite reading for a linear axis (activations, entropy): non-finite → null.
const fin = (v: number): number | null => (Number.isFinite(v) ? v : null)

// Compact axis/tooltip number: scientific for the very small/large (grad norms,
// ratios span decades), otherwise a short fixed form.
function sci(v: number): string {
  if (!Number.isFinite(v)) return "—"
  const a = Math.abs(v)
  if (a !== 0 && (a < 0.01 || a >= 1000)) return v.toExponential(0)
  return a < 1 ? v.toFixed(2) : v.toFixed(1)
}

const stepLabel = (v: React.ReactNode) => `Step ${v}`

// A small early→late colour-ramp legend, mirroring the loss overlay's legend so
// the per-layer figures read as one family. `label` names what the ramp indexes.
function RampLegend({
  from,
  to,
  label,
}: {
  from: string
  to: string
  label: string
}) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="text-muted-foreground">{label} 0</span>
      <span
        aria-hidden
        className="h-1.5 w-16 rounded-full"
        style={{ background: `linear-gradient(to right, ${from}, ${to})` }}
      />
      <span className="text-muted-foreground">n</span>
    </span>
  )
}

// One trace in a figure.
interface Trace {
  key: string
  name: string
  color: string
  width?: number
  dash?: string
}

// The shared multi-line chart for the per-layer / per-head figures. Kept generic
// (traces + rows + axis mode) so the grad-norm, update-ratio, activation and
// attention figures all render identically — only their data and y-scale differ.
function TraceChart({
  rows,
  traces,
  log,
  domain,
  reference,
}: {
  rows: Record<string, number | null>[]
  traces: Trace[]
  log?: boolean
  domain?: [number | string, number | string]
  reference?: { y: number; label: string }
}) {
  const config = React.useMemo(
    () =>
      Object.fromEntries(
        traces.map((t) => [t.key, { label: t.name, color: t.color }])
      ) satisfies ChartConfig,
    [traces]
  )

  return (
    <ChartContainer config={config} className="aspect-auto h-64 w-full">
      <ComposedChart data={rows} margin={{ left: 4, right: 4, top: 4 }}>
        <CartesianGrid {...gridProps} />
        <XAxis
          dataKey="step"
          type="number"
          domain={["dataMin", "dataMax"]}
          {...axisProps}
        />
        <YAxis
          {...yAxisProps}
          scale={log ? "log" : "auto"}
          domain={domain ?? ["auto", "auto"]}
          allowDataOverflow
          tickFormatter={sci}
        />
        <ChartTooltip
          cursor={false}
          content={<ChartTooltipContent labelFormatter={stepLabel} />}
        />
        {reference && (
          <ReferenceLine
            y={reference.y}
            stroke={ACCENT_COLOR}
            strokeDasharray="4 4"
            strokeOpacity={0.7}
            label={{
              value: reference.label,
              position: "insideTopRight",
              fontSize: 10,
              fill: "var(--muted-foreground)",
            }}
          />
        )}
        {traces.map((t) => (
          <Line
            key={t.key}
            dataKey={t.key}
            name={t.name}
            type="monotone"
            stroke={t.color}
            strokeWidth={t.width ?? 1.5}
            strokeDasharray={t.dash}
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
        ))}
      </ComposedChart>
    </ChartContainer>
  )
}

export interface DiagnosticsPanelProps {
  diag: DiagPoint[]
}

// The training-diagnostics spread: a tabbed panel below the loss figures. Each
// tab is one metric family, rendered in the same captioned-Figure idiom as the
// loss/spread charts so the page reads as one editorial system. The section keeps
// its slot from first paint and shows a placeholder until the first report lands.
export function DiagnosticsPanel({ diag }: DiagnosticsPanelProps) {
  const last = diag.at(-1)
  const layerCount = last?.grad_layers.length ?? 0
  const headCount = last?.attn_entropy[0]?.length ?? 0

  const [attnLayer, setAttnLayer] = React.useState(0)
  // Clamp at read time (not via an effect) so a new run with fewer layers can't
  // leave the selector pointing past the end.
  const selLayer = attnLayer < layerCount ? attnLayer : 0

  const layerTraces = React.useMemo<Trace[]>(
    () =>
      Array.from({ length: layerCount }, (_, i) => ({
        key: `l${i}`,
        name: `Layer ${i}`,
        color: layerColor(i, layerCount),
      })),
    [layerCount]
  )

  // ── grad norms: per-layer ramp + global accent + the two shared tables ──────
  const gradRows = React.useMemo(
    () =>
      diag.map((d) => {
        const row: Record<string, number | null> = {
          step: d.step,
          global: pos(d.grad_global),
          embedding: pos(d.grad_embedding),
          unembed: pos(d.grad_unembed),
        }
        d.grad_layers.forEach((v, i) => (row[`l${i}`] = pos(v)))
        return row
      }),
    [diag]
  )
  const gradTraces: Trace[] = [
    ...layerTraces,
    { key: "global", name: "Global", color: ACCENT_COLOR, width: 2 },
    { key: "embedding", name: "Embedding", color: MUTED_SERIES, dash: "3 3" },
    { key: "unembed", name: "Unembed", color: MUTED_SERIES, dash: "1 3" },
  ]

  // ── update:weight ratio ─────────────────────────────────────────────────────
  const updateRows = React.useMemo(
    () =>
      diag.map((d) => {
        const row: Record<string, number | null> = { step: d.step }
        d.update_ratio_layers.forEach((v, i) => (row[`l${i}`] = pos(v)))
        return row
      }),
    [diag]
  )

  // ── activation RMS per block ────────────────────────────────────────────────
  const actRows = React.useMemo(
    () =>
      diag.map((d) => {
        const row: Record<string, number | null> = { step: d.step }
        d.act_rms_layers.forEach((v, i) => (row[`l${i}`] = fin(v)))
        return row
      }),
    [diag]
  )

  // ── attention entropy for the selected layer ────────────────────────────────
  const attnTraces = React.useMemo<Trace[]>(
    () =>
      Array.from({ length: headCount }, (_, j) => ({
        key: `h${j}`,
        name: `Head ${j}`,
        color: headColor(j, headCount),
      })),
    [headCount]
  )
  const attnRows = React.useMemo(
    () =>
      diag.map((d) => {
        const row: Record<string, number | null> = { step: d.step }
        const heads = d.attn_entropy[selLayer] ?? []
        heads.forEach((v, j) => (row[`h${j}`] = fin(v)))
        return row
      }),
    [diag, selLayer]
  )

  // ── embedding anisotropy: mean line + p5–p95 band ───────────────────────────
  const embRows = React.useMemo(
    () =>
      diag.map((d) => ({
        step: d.step,
        mean: fin(d.emb_cos_mean),
        band: [fin(d.emb_cos_p5), fin(d.emb_cos_p95)] as [
          number | null,
          number | null,
        ],
      })),
    [diag]
  )
  const embConfig = {
    mean: { label: "Mean cosine", color: EMB_COLOR },
  } satisfies ChartConfig

  const ready = diag.length > 1
  const ramp = (
    <RampLegend
      from={layerColor(0, layerCount)}
      to={layerColor(layerCount - 1, layerCount)}
      label="layer"
    />
  )

  return (
    <section className="space-y-3 border-t pt-6">
      <div className="flex items-baseline justify-between">
        <h2 className="font-heading text-sm font-semibold tracking-tight">
          Diagnostics
        </h2>
        <span className="text-xs text-muted-foreground">
          sampled per report ({diag.length})
        </span>
      </div>

      {!ready ? (
        <FigurePlaceholder className="h-64">
          Waiting for the first diagnostics sample…
        </FigurePlaceholder>
      ) : (
        <Tabs defaultValue="grad">
          <TabsList>
            <TabsTrigger value="grad">Grad norms</TabsTrigger>
            <TabsTrigger value="update">Update ratio</TabsTrigger>
            <TabsTrigger value="act">Activations</TabsTrigger>
            <TabsTrigger value="attn">Attention</TabsTrigger>
            <TabsTrigger value="emb">Embeddings</TabsTrigger>
          </TabsList>

          <TabsContent value="grad">
            <Figure
              caption="Gradient L2 norm — per layer, global, and the shared tables (log scale)"
              aside={ramp}
            >
              <TraceChart rows={gradRows} traces={gradTraces} log />
            </Figure>
          </TabsContent>

          <TabsContent value="update">
            <Figure
              caption="Update : weight ratio ‖Δw‖/‖w‖ per layer (log scale) — ~1e-3 is healthy"
              aside={ramp}
            >
              <TraceChart
                rows={updateRows}
                traces={layerTraces}
                log
                reference={{ y: 1e-3, label: "1e-3" }}
              />
            </Figure>
          </TabsContent>

          <TabsContent value="act">
            <Figure
              caption="Activation RMS of the residual stream entering each block"
              aside={ramp}
            >
              <TraceChart
                rows={actRows}
                traces={layerTraces}
                domain={[0, "auto"]}
              />
            </Figure>
          </TabsContent>

          <TabsContent value="attn">
            <Figure
              caption="Attention entropy per head (normalized: 1 = diffuse, 0 = sharp)"
              aside={
                <div className="flex items-center gap-2">
                  <RampLegend
                    from={headColor(0, headCount)}
                    to={headColor(headCount - 1, headCount)}
                    label="head"
                  />
                  <Select
                    value={String(selLayer)}
                    onValueChange={(v) => setAttnLayer(Number(v))}
                  >
                    <SelectTrigger size="sm" className="h-7 w-28">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {Array.from({ length: layerCount }, (_, i) => (
                        <SelectItem key={i} value={String(i)}>
                          Layer {i}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              }
            >
              <TraceChart
                rows={attnRows}
                traces={attnTraces}
                domain={[0, 1]}
              />
            </Figure>
          </TabsContent>

          <TabsContent value="emb">
            <Figure caption="Embedding anisotropy — pairwise cosine among frequent tokens (mean · p5–p95 band)">
              <ChartContainer
                config={embConfig}
                className="aspect-auto h-64 w-full"
              >
                <ComposedChart
                  data={embRows}
                  margin={{ left: 4, right: 4, top: 4 }}
                >
                  <CartesianGrid {...gridProps} />
                  <XAxis
                    dataKey="step"
                    type="number"
                    domain={["dataMin", "dataMax"]}
                    {...axisProps}
                  />
                  <YAxis
                    {...yAxisProps}
                    domain={["auto", "auto"]}
                    tickFormatter={(v: number) => v.toFixed(2)}
                  />
                  <ChartTooltip
                    cursor={false}
                    content={<ChartTooltipContent labelFormatter={stepLabel} />}
                  />
                  <Area
                    dataKey="band"
                    name="p5–p95"
                    type="monotone"
                    stroke="none"
                    fill={EMB_COLOR}
                    fillOpacity={0.15}
                    isAnimationActive={false}
                    connectNulls
                  />
                  <Line
                    dataKey="mean"
                    name="Mean cosine"
                    type="monotone"
                    stroke={EMB_COLOR}
                    strokeWidth={1.5}
                    dot={false}
                    connectNulls
                    isAnimationActive={false}
                  />
                </ComposedChart>
              </ChartContainer>
            </Figure>
          </TabsContent>
        </Tabs>
      )}
    </section>
  )
}
