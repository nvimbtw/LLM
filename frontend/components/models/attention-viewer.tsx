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
import { Textarea } from "@/components/ui/textarea"
import { useAttention } from "@/hooks/use-attention"
import type { AttentionCell } from "@/lib/attention"
import { AttentionCanvas } from "./attention-canvas"

export interface AttentionViewerProps {
  modelId: string
  layers: number
  heads: number
}

// Long enough that the matrix has structure to look at, short enough to read.
const SAMPLE_TEXT =
  "The cat sat on the mat. The cat ate the fish, and the dog watched it."

// Two decimals says nothing about a weight of 0.0004, and a diffuse head is
// mostly weights of that size.
const WEIGHT_DIGITS = 4
const MILLISECOND_DIGITS = 1

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

function IndexSelect({
  label,
  count,
  value,
  onChange,
  disabled,
}: {
  label: string
  count: number
  value: number
  onChange: (next: number) => void
  disabled: boolean
}) {
  return (
    <Field label={label}>
      <Select
        value={String(value)}
        // A Select hands back `null` for "nothing chosen"; there is no such
        // state here, so it falls back to what is already selected.
        onValueChange={(next) => onChange(Number(next ?? value))}
        disabled={disabled}
      >
        <SelectTrigger className="w-24">
          {/* Bare, `<SelectValue />` renders the raw value — the trigger would
              read `0` while the list it opens reads `Layer 0`. */}
          <SelectValue>{(chosen: string | null) => chosen ?? "0"}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {Array.from({ length: count }, (_, index) => (
            <SelectItem key={index} value={String(index)}>
              {index}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </Field>
  )
}

// A token is not prose: it can be a leading space, a newline, or half of a
// multi-byte character, and all three have to be visible as what they are.
function TokenChip({ token }: { token: string }) {
  return (
    <span className="tag rounded-(--radius-marker) bg-muted px-1 py-0.5 whitespace-pre text-foreground">
      {token}
    </span>
  )
}

export function AttentionViewer({
  modelId,
  layers,
  heads,
}: AttentionViewerProps) {
  const { matrix, isLoading, error, load } = useAttention(modelId)
  const [text, setText] = React.useState(SAMPLE_TEXT)
  const [layer, setLayer] = React.useState(0)
  const [head, setHead] = React.useState(0)
  const [cell, setCell] = React.useState<AttentionCell | null>(null)
  const [drawMs, setDrawMs] = React.useState<number | null>(null)

  function show() {
    setCell(null)
    void load({ text, layer, head })
  }

  return (
    <Panel
      title="Attention"
      aside={
        drawMs !== null &&
        matrix && (
          // The honest number, on the reader's own machine. Whether canvas2d is
          // enough for this figure is a question that should be settled by
          // measurement rather than by anyone's prior about canvas.
          <span className="text-xs text-muted-foreground tabular-nums">
            {matrix.size}×{matrix.size} drawn in{" "}
            {drawMs.toFixed(MILLISECOND_DIGITS)} ms
          </span>
        )
      }
    >
      <div className="flex flex-col gap-4">
        <p className="max-w-prose text-sm text-muted-foreground">
          Nothing records attention while a model trains or answers — the fused
          kernel never builds the matrix. This asks for exactly one head&apos;s,
          over exactly this text.
        </p>

        <div className="flex flex-wrap items-end gap-3">
          <IndexSelect
            label="Layer"
            count={layers}
            value={layer}
            onChange={setLayer}
            disabled={isLoading}
          />
          <IndexSelect
            label="Head"
            count={heads}
            value={head}
            onChange={setHead}
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

        <Field label="Text">
          <Textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
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
                // A `useState` setter is identity-stable, so the canvas' draw
                // effect fires on a new matrix and not on every hover.
                onCellChange={setCell}
                onDrawn={setDrawMs}
              />
            </div>

            {/* Fixed height: a readout that appears on hover and disappears on
                leave would reflow the figure out from under the pointer. */}
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
