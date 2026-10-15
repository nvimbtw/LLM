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

/** Offers the available zero-based layer or head indices using descriptive option labels. */
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
        onValueChange={(next) => onChange(Number(next ?? value))}
        disabled={disabled}
      >
        <SelectTrigger className="w-24">
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

/** Displays one token in the selected attention-cell readout, preserving significant spaces and newlines. */
function TokenChip({ token }: { token: string }) {
  return (
    <span className="rounded-(--radius-marker) bg-muted px-1 py-0.5 tag whitespace-pre text-foreground">
      {token}
    </span>
  )
}

/** Requests and displays one selected attention head over the supplied text, preserving loading and failure feedback. */
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
