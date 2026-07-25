"use client"

import * as React from "react"
import { Pause, Play } from "lucide-react"
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts"

import { Button } from "@/components/ui/button"
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
import { Slider } from "@/components/ui/slider"
import {
  ACCENT_COLOR,
  axisProps,
  EMB_COLOR,
  embRankColor,
  gridProps,
  headColor,
  layerColor,
  MUTED_SERIES,
  yAxisProps,
} from "@/components/training/chart-theme"
import { EmbeddingGlobe } from "@/components/training/embedding-globe"
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

// Evenly sample `arr` down to at most `max` items (keeps the first and last), for
// the heatmap columns — a long run would otherwise mean hundreds of 1px columns.
function sample<T>(arr: T[], max: number): T[] {
  if (arr.length <= max) return arr
  const out: T[] = []
  const stride = (arr.length - 1) / (max - 1)
  for (let i = 0; i < max; i++) out.push(arr[Math.round(i * stride)])
  return out
}

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
            type="linear"
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

// Distribution-over-time heatmap: a bin×time grid where each column is one report's
// z-scored activation histogram (top = +range σ, bottom = −range σ) and cell
// opacity encodes density. A CSS grid rather than a charting primitive — it's the
// cleanest way to render a dense heatmap responsively. Columns are sampled so the
// grid stays legible over a long run.
function ActivationHeatmap({ diag }: { diag: DiagPoint[] }) {
  const cols = React.useMemo(
    () => sample(diag.filter((d) => d.act_hist?.length), 64),
    [diag]
  )
  const bins = cols[0]?.act_hist.length ?? 0
  const max = React.useMemo(() => {
    let m = 0
    for (const c of cols) for (const v of c.act_hist) if (v > m) m = v
    return m || 1
  }, [cols])

  // Row-major, top row = highest σ bin. A gentle gamma lifts mid densities so the
  // distribution's shoulders stay visible, not just its peak.
  const cells = React.useMemo(() => {
    const out: { alpha: number }[] = []
    for (let row = 0; row < bins; row++) {
      const binIndex = bins - 1 - row
      for (const c of cols) {
        const alpha = Math.pow((c.act_hist[binIndex] ?? 0) / max, 0.65)
        out.push({ alpha })
      }
    }
    return out
  }, [cols, bins, max])

  if (bins === 0) {
    return <FigurePlaceholder className="h-64">No histogram yet…</FigurePlaceholder>
  }

  return (
    <div className="flex h-64 gap-2">
      <div className="flex flex-col justify-between py-0.5 text-[10px] text-muted-foreground tabular-nums">
        <span>+4σ</span>
        <span>0</span>
        <span>−4σ</span>
      </div>
      <div
        className="grid flex-1 overflow-hidden rounded-md border"
        style={{
          gridTemplateColumns: `repeat(${cols.length}, 1fr)`,
          gridTemplateRows: `repeat(${bins}, 1fr)`,
        }}
      >
        {cells.map((c, i) => (
          <div
            key={i}
            style={{ background: `hsl(168 55% 45% / ${c.alpha})` }}
          />
        ))}
      </div>
    </div>
  )
}

type Pt = [number, number]

// Sign-align each PCA frame to the previous one. Eigenvectors have an arbitrary
// sign, so consecutive reports can mirror on either axis; flipping each axis to
// best match the prior frame keeps the arrows from flipping as you scrub or play.
function alignFrames(frames: Pt[][]): Pt[][] {
  const out: Pt[][] = []
  let ref: Pt[] | null = null
  for (const f of frames) {
    if (!ref || ref.length !== f.length) {
      out.push(f)
      ref = f
      continue
    }
    let dotX = 0
    let dotY = 0
    for (let i = 0; i < f.length; i++) {
      dotX += ref[i][0] * f[i][0]
      dotY += ref[i][1] * f[i][1]
    }
    const sx = dotX >= 0 ? 1 : -1
    const sy = dotY >= 0 ? 1 : -1
    const a: Pt[] = f.map(([x, y]) => [x * sx, y * sy])
    out.push(a)
    ref = a
  }
  return out
}

const easeInOutCubic = (p: number): number =>
  p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2

// Embedding PCA biplot: frequent-token embeddings drawn as vectors from the centre
// (the mean embedding) in the plane of the top-2 principal components, with a
// time-scrubber to play through every report. Because PCA scores are mean-centred,
// the origin IS the average token, so an arrow's direction is a token's embedding
// orientation, its length is how far it sits from the average, and the angle between
// two arrows tracks their similarity. Hand-rolled square SVG (recharts' scatter can't
// draw origin-anchored arrows); the arrows tween between frames so the geometry
// morphs smoothly. PC axes are arbitrary in sign/scale (labelled, not ticked).
function EmbeddingBiplot({ frames, steps }: { frames: Pt[][]; steps: number[] }) {
  const aligned = React.useMemo(() => alignFrames(frames), [frames])
  // One radius across all frames so the axes don't rescale mid-animation — the
  // arrows visibly grow as the embeddings pull away from the mean over training.
  const radius = React.useMemo(() => {
    let r = 0
    for (const f of aligned)
      for (const [x, y] of f) r = Math.max(r, Math.abs(x), Math.abs(y))
    return (r || 1) * 1.12
  }, [aligned])

  const lastIndex = Math.max(0, aligned.length - 1)
  // `pinned` is where the user parked the scrubber; `live` means "track the newest
  // report". The shown frame is derived, so following the live edge needs no effect.
  const [pinned, setPinned] = React.useState(lastIndex)
  const [live, setLive] = React.useState(true)
  const [playing, setPlaying] = React.useState(false)
  const idx = live ? lastIndex : Math.min(pinned, lastIndex)

  // Displayed coordinates in data space, tweened toward frame `idx` each time it
  // changes (via scrub, playback, or a new live report).
  const [disp, setDisp] = React.useState<Pt[]>(aligned[lastIndex] ?? [])
  const dispRef = React.useRef<Pt[]>(disp)

  React.useEffect(() => {
    const target = aligned[idx] ?? []
    const start = dispRef.current
    const t0 = performance.now()
    const dur = 280
    let raf = 0
    const tick = (now: number) => {
      // A new run can change the token count — snap rather than interpolate.
      if (start.length !== target.length) {
        dispRef.current = target
        setDisp(target)
        return
      }
      const e = easeInOutCubic(Math.min(1, (now - t0) / dur))
      const next: Pt[] = target.map(([tx, ty], i) => {
        const [sx, sy] = start[i]
        return [sx + (tx - sx) * e, sy + (ty - sy) * e]
      })
      dispRef.current = next
      setDisp(next)
      if (e < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [idx, aligned])

  // Playback: advance a frame at a time, then stop and re-attach to the live edge.
  React.useEffect(() => {
    if (!playing) return
    const id = setInterval(() => {
      setPinned((prev) => {
        if (prev + 1 >= lastIndex) {
          setPlaying(false)
          setLive(true)
          return lastIndex
        }
        return prev + 1
      })
    }, 180)
    return () => clearInterval(id)
  }, [playing, lastIndex])

  if (aligned.length === 0 || (aligned[idx]?.length ?? 0) < 2) {
    return (
      <FigurePlaceholder className="mx-auto aspect-square w-full max-w-[440px]">
        Waiting for the embedding projection…
      </FigurePlaceholder>
    )
  }

  const S = 100
  const M = 10
  const C = S / 2
  const half = (S - 2 * M) / 2
  const toX = (x: number) => C + (x / radius) * half
  const toY = (y: number) => C - (y / radius) * half
  const n = disp.length

  const scrub = (v: number | readonly number[]) => {
    const value = Array.isArray(v) ? (v[0] ?? lastIndex) : (v as number)
    setPlaying(false)
    setPinned(value)
    setLive(value >= lastIndex)
  }

  const togglePlay = () => {
    if (playing) {
      setPlaying(false)
    } else {
      // Replay from the start if we're parked at the live edge, else resume.
      if (idx >= lastIndex) {
        setPinned(0)
        setLive(false)
      }
      setPlaying(true)
    }
  }

  return (
    <div className="mx-auto w-full max-w-[440px] space-y-3">
      <svg
        viewBox={`0 0 ${S} ${S}`}
        preserveAspectRatio="xMidYMid meet"
        className="block aspect-square w-full rounded-md border"
      >
        <line x1={M} y1={C} x2={S - M} y2={C} stroke="var(--border)" strokeWidth={0.4} />
        <line x1={C} y1={M} x2={C} y2={S - M} stroke="var(--border)" strokeWidth={0.4} />
        <text x={S - M} y={C - 1.6} textAnchor="end" fontSize={3.4} fill="var(--muted-foreground)">
          PC1 →
        </text>
        <text x={C + 1.6} y={M + 3.2} textAnchor="start" fontSize={3.4} fill="var(--muted-foreground)">
          PC2 ↑
        </text>

        {disp.map(([x, y], i) => {
          const px = toX(x)
          const py = toY(y)
          const dx = px - C
          const dy = py - C
          const len = Math.hypot(dx, dy)
          const color = embRankColor(i, n)
          let head = ""
          if (len > 0.5) {
            const ux = dx / len
            const uy = dy / len
            const a = 2.4
            const w = 1.4
            const bx = px - a * ux
            const by = py - a * uy
            head = `M ${px} ${py} L ${bx + w * -uy} ${by + w * ux} L ${bx - w * -uy} ${by - w * ux} Z`
          }
          return (
            <g key={i}>
              <line
                x1={C}
                y1={C}
                x2={px}
                y2={py}
                stroke={color}
                strokeWidth={0.8}
                strokeOpacity={0.85}
                strokeLinecap="round"
              />
              {head && <path d={head} fill={color} fillOpacity={0.9} />}
            </g>
          )
        })}
        <circle cx={C} cy={C} r={0.9} fill="var(--muted-foreground)" />
      </svg>

      {/* Time scrubber: play through the reports, or drag to a moment in training. */}
      <div className="flex items-center gap-3">
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="size-7 shrink-0"
          onClick={togglePlay}
          disabled={lastIndex === 0}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
        </Button>
        <Slider
          value={[idx]}
          min={0}
          max={lastIndex}
          step={1}
          onValueChange={scrub}
          disabled={lastIndex === 0}
          className="flex-1"
          aria-label="Training report"
        />
        <span className="w-24 shrink-0 text-right text-xs text-muted-foreground tabular-nums">
          step {steps[idx] ?? 0}
          {live ? " · live" : ""}
        </span>
      </div>
    </div>
  )
}

export interface DiagnosticsPanelProps {
  diag: DiagPoint[]
}

// The training-diagnostics deep-dive: a full-width stack of figures below the
// rail+hero section. Each metric is one captioned figure in the same idiom as the
// loss/spread charts, so the page reads as one editorial system. The section keeps
// its slot from first paint and shows a placeholder until the first report lands.
export function DiagnosticsPanel({ diag }: DiagnosticsPanelProps) {
  const last = diag.at(-1)
  const layerCount = last?.grad_layers.length ?? 0
  const headCount = last?.attn_entropy[0]?.length ?? 0
  const pcaCount = last?.emb_pca.length ?? 0

  const [attnLayer, setAttnLayer] = React.useState(0)
  // Clamp at read time (not via an effect) so a new run with fewer layers can't
  // leave the selector pointing past the end.
  const selLayer = attnLayer < layerCount ? attnLayer : 0

  // Small derived arrays (a handful of layers/heads) — left as plain consts for the
  // React Compiler to memoize; a manual useMemo here can't be preserved once they
  // feed the composed trace lists below.
  const layerTraces: Trace[] = Array.from({ length: layerCount }, (_, i) => ({
    key: `l${i}`,
    name: `Layer ${i}`,
    color: layerColor(i, layerCount),
  }))

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
  const attnTraces: Trace[] = Array.from({ length: headCount }, (_, j) => ({
    key: `h${j}`,
    name: `Head ${j}`,
    color: headColor(j, headCount),
  }))
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
    <section className="space-y-6 border-t pt-6">
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
        <div className="space-y-8">
          <Figure
            caption="Gradient L2 norm — per layer, global, and the shared tables (log scale)"
            aside={ramp}
          >
            <TraceChart rows={gradRows} traces={gradTraces} log />
          </Figure>

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

          <Figure
            caption="Activation RMS of the residual stream entering each block"
            aside={ramp}
          >
            <TraceChart rows={actRows} traces={layerTraces} domain={[0, "auto"]} />
          </Figure>

          <Figure caption="Activation distribution over time — deepest residual stream, z-scored (colour = density)">
            <ActivationHeatmap diag={diag} />
          </Figure>

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
            <TraceChart rows={attnRows} traces={attnTraces} domain={[0, 1]} />
          </Figure>

          <Figure caption="Embedding anisotropy — pairwise cosine among frequent tokens (mean · p5–p95 band)">
            <ChartContainer
              config={embConfig}
              className="aspect-auto h-64 w-full"
            >
              <ComposedChart data={embRows} margin={{ left: 4, right: 4, top: 4 }}>
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
                  type="linear"
                  stroke="none"
                  fill={EMB_COLOR}
                  fillOpacity={0.15}
                  isAnimationActive={false}
                  connectNulls
                />
                <Line
                  dataKey="mean"
                  name="Mean cosine"
                  type="linear"
                  stroke={EMB_COLOR}
                  strokeWidth={1.5}
                  dot={false}
                  connectNulls
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ChartContainer>
          </Figure>

          <Figure
            caption="Embedding geometry — frequent tokens on the top principal axes (latest report)"
            aside={
              <RampLegend
                from={embRankColor(0, pcaCount)}
                to={embRankColor(pcaCount - 1, pcaCount)}
                label="rank"
              />
            }
          >
            {/* Two lenses on the same PCA cloud: the flat top-2 biplot, and the
                rotating top-3 solid. The biplot is square, so they sit side by side
                on wide screens and stack on narrow ones. */}
            <div className="grid gap-8 lg:grid-cols-2 lg:items-start">
              <div>
                <EmbeddingBiplot
                  frames={diag.map((d) => d.emb_pca)}
                  steps={diag.map((d) => d.step)}
                />
                <p className="mx-auto mt-3 max-w-[440px] text-xs leading-relaxed text-muted-foreground">
                  Each arrow runs from the{" "}
                  <span className="text-foreground">mean embedding</span> (centre) to
                  a frequent token, in the plane of the top-2 principal components.{" "}
                  <span className="text-foreground">Direction</span> is the
                  token&apos;s embedding orientation,{" "}
                  <span className="text-foreground">length</span> is how far it sits
                  from the average, and the{" "}
                  <span className="text-foreground">angle</span> between two arrows
                  tracks their similarity. PC1 (→) and PC2 (↑) are the directions of
                  greatest variance; their sign and scale are arbitrary.
                </p>
              </div>
              <div>
                <EmbeddingGlobe frames={diag.map((d) => d.emb_pca3)} variant="full" />
                <p className="mx-auto mt-3 max-w-[440px] text-xs leading-relaxed text-muted-foreground">
                  The same tokens as an{" "}
                  <span className="text-foreground">icosphere</span>, adding{" "}
                  <span className="text-foreground">PC3</span> as depth — each token
                  sits on the sphere in the{" "}
                  <span className="text-foreground">direction</span> its embedding
                  points, and the surface is triangulated between them. Tokens that
                  point alike sit close together; the gentle bulges track how far each
                  sits from the average. Drag to orbit, or let it turn.
                </p>
              </div>
            </div>
          </Figure>
        </div>
      )}
    </section>
  )
}
