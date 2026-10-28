"use client"

import * as React from "react"

import { Button } from "@/components/ui/button"
import { FormError } from "@/components/ui/error-state"
import { Panel } from "@/components/ui/panel"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { useAttention } from "@/hooks/use-attention"
import type { AttentionCell } from "@/lib/attention"
import { AttentionCanvas } from "./attention-canvas"

export interface AttentionViewerProps {
  modelId: string
}

const SAMPLE_TEXT =
  "The cat sat on the mat. The cat ate the fish, and the dog watched it."

const WEIGHT_DIGITS = 4
const MILLISECOND_DIGITS = 1

/** Labels one attention control or input. */
function Field({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="eyebrow text-muted-foreground">{label}</span>
      {children}
    </label>
  )
}

/** Edits one bounded index using actual dimensions without allocating an option per possible layer or head. */
function IndexInput({
  label,
  count,
  value,
  onChange,
  disabled,
}: {
  label: string
  count: number | null
  value: number
  onChange: (next: number) => void
  disabled: boolean
}) {
  return (
    <Field label={label}>
      <Input
        type="number"
        className="w-24"
        min={0}
        max={count == null ? 255 : count - 1}
        step={1}
        value={value}
        disabled={disabled}
        onChange={(event) => {
          const next = Number(event.target.value)
          if (Number.isSafeInteger(next) && next >= 0 && next < (count ?? 256))
            onChange(next)
        }}
      />
    </Field>
  )
}

/** Displays one token in the selected attention-cell readout, preserving significant spaces and newlines. */
function TokenChip({ token }: { token: string }) {
  return (
    <span className="rounded-(--radius-marker) bg-muted px-1 py-0.5 tag whitespace-pre text-foreground">
      {token}
    </span>
  )
}

/** Requests and displays one selected attention head over the supplied text, preserving loading and failure feedback. */
export function AttentionViewer({ modelId }: AttentionViewerProps) {
  const { catalog, matrix, isLoading, error, discover, load, clear } =
    useAttention(modelId)
  const [text, setText] = React.useState(SAMPLE_TEXT)
  const [layer, setLayer] = React.useState(0)
  const [head, setHead] = React.useState(0)
  const [pointId, setPointId] = React.useState<string | null>(null)
  const [cell, setCell] = React.useState<AttentionCell | null>(null)
  const [drawMs, setDrawMs] = React.useState<number | null>(null)

  const points = catalog?.points ?? []
  const selectedPoint =
    points.find((point) => point.id === pointId) ?? points[0]
  const canInspect = Boolean(selectedPoint || catalog?.legacy_layers)
  const heads = selectedPoint?.heads ?? catalog?.legacy_heads ?? null

  /** Removes the previous figure and pending request when text or selection changes. */
  function reset() {
    clear()
    setCell(null)
    setDrawMs(null)
  }

  /** Requests the actual selected occurrence or the compatible legacy layer with no invented generic dimensions. */
  function show() {
    setCell(null)
    void load(
      selectedPoint
        ? { text, point_id: selectedPoint.id, head }
        : { text, layer, head }
    )
  }

  return (
    <Panel
      variant="open"
      title="Attention"
      aside={
        drawMs !== null &&
        matrix && (
          <span className="text-xs text-muted-foreground tabular-nums">
            {matrix.size}×{matrix.size} drawn in{" "}
            {drawMs.toFixed(MILLISECOND_DIGITS)} ms
          </span>
        )
      }
    >
      <div className="flex flex-col gap-4">
        <p className="max-w-prose text-sm text-muted-foreground">
          Inspect one attention head over your text. Finding views loads the
          selected model when needed.
        </p>

        {!catalog && (
          <Button
            size="sm"
            className="self-start"
            disabled={isLoading}
            onClick={() => void discover()}
          >
            {isLoading ? "Finding views…" : "Find attention views"}
          </Button>
        )}
        {catalog?.unavailable_reason && (
          <p className="text-sm text-muted-foreground">
            {catalog.unavailable_reason}
          </p>
        )}
        {canInspect && (
          <div className="flex flex-wrap items-end gap-3">
            {selectedPoint ? (
              <Field label="Attention view">
                <Select
                  value={selectedPoint.id}
                  disabled={isLoading}
                  onValueChange={(value) => {
                    if (!value) return
                    reset()
                    setPointId(value)
                    setHead(0)
                  }}
                >
                  <SelectTrigger className="max-w-full sm:w-96">
                    <SelectValue>{selectedPoint.title}</SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {points.map((point) => (
                      <SelectItem key={point.id} value={point.id}>
                        {point.title} · {point.instance_path} · call{" "}
                        {point.call_path.join(".") || "0"}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
            ) : (
              <IndexInput
                label="Layer"
                count={catalog?.legacy_layers ?? null}
                value={layer}
                onChange={(value) => {
                  reset()
                  setLayer(value)
                }}
                disabled={isLoading}
              />
            )}
            <IndexInput
              label="Head"
              count={heads}
              value={head}
              onChange={(value) => {
                reset()
                setHead(value)
              }}
              disabled={isLoading}
            />
            <Button
              size="sm"
              onClick={show}
              disabled={isLoading || text.trim().length === 0}
            >
              {isLoading ? "Reading…" : "Show attention"}
            </Button>
          </div>
        )}
        {isLoading && (
          <Button
            size="sm"
            variant="outline"
            className="self-start"
            onClick={reset}
          >
            Cancel
          </Button>
        )}
        {selectedPoint && (
          <p className="text-xs text-muted-foreground">
            {selectedPoint.instance_path} · call{" "}
            {selectedPoint.call_path.join(".") || "0"}
            {selectedPoint.validated_sequence_max
              ? ` · Validated through ${selectedPoint.validated_sequence_max} tokens`
              : " · Checked when inspected"}
            {heads == null
              ? " · Head count checked when inspected"
              : ` · ${heads} heads`}
          </p>
        )}

        <Field label="Text">
          <Textarea
            value={text}
            onChange={(event) => {
              reset()
              setText(event.target.value)
            }}
            maxLength={1048576}
            rows={3}
            spellCheck={false}
          />
        </Field>

        {error && <FormError>{error}</FormError>}

        {matrix && (
          <div className="flex flex-col gap-3">
            <div className="mx-auto w-full max-w-xl">
              <AttentionCanvas
                matrix={matrix}
                cell={cell}
                onCellChange={setCell}
                onDrawn={setDrawMs}
              />
            </div>

            <div className="flex min-h-9 flex-wrap items-center justify-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
              {cell ? (
                <>
                  <TokenChip token={matrix.tokens[cell.query]} />
                  <span>({cell.query}) attended to</span>
                  <TokenChip token={matrix.tokens[cell.key]} />
                  <span>({cell.key}) at</span>
                  <span className="figure text-foreground tabular-nums">
                    {cell.weight.toFixed(WEIGHT_DIGITS)}
                  </span>
                </>
              ) : (
                <span>
                  Rows are query positions, columns are keys. Hover a cell.
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </Panel>
  )
}
