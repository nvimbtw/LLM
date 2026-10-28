"use client"

import { useRouter } from "next/navigation"
import Link from "next/link"
import * as React from "react"
import { ChevronDown, ChevronRight, Plus, X } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { FormError, LoadError } from "@/components/ui/error-state"
import { Input } from "@/components/ui/input"
import { Panel } from "@/components/ui/panel"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { toast } from "@/components/ui/toast"
import { ProcessorSource } from "@/components/data/processor-source"
import { useSessionToken } from "@/hooks/use-auth"
import { useProcessingWorkbench } from "@/hooks/use-processing-workbench"
import {
  documentText,
  processDataset,
  type Dataset,
  type ParamValue,
  type PipelineStep,
  type Preview,
  type Processor,
} from "@/lib/data"
import { hunks, splitDiff, type DiffRow } from "@/lib/diff"
import { cn } from "@/lib/utils"

export interface ProcessingWorkbenchProps {
  dataset: Dataset
}

/**
 * Pick cleaning steps and watch what they do, before committing to a run.
 *
 * The preview is the point: choosing a step by reading its description is
 * guesswork. It always shows the *first* documents by ordinal, so nudging a
 * parameter compares two runs over the same text rather than over a new sample.
 */
export function ProcessingWorkbench({ dataset }: ProcessingWorkbenchProps) {
  const token = useSessionToken()
  return (
    <Workbench
      key={`${dataset.id}:${token ?? "anonymous"}`}
      dataset={dataset}
    />
  )
}

/** Presents one dataset/session's pipeline and overall preview with source controls. */
function Workbench({ dataset }: ProcessingWorkbenchProps) {
  const router = useRouter()
  const {
    processors,
    isLoading,
    error,
    steps,
    preview,
    previewError,
    unused,
    add,
    remove,
    update,
    sourceChanged,
    sampleSize,
  } = useProcessingWorkbench(dataset)
  const [naming, setNaming] = React.useState(false)
  const [opened, setOpened] = React.useState<string | null>(null)
  const openIndex = steps.findIndex((s) => s.processor === opened)
  const openStep = openIndex === -1 ? null : steps[openIndex]
  const openProcessor = processors.find((p) => p.name === opened) ?? null

  if (error) {
    return <LoadError title="The processors didn't load." hint={error} />
  }

  return (
    <div className="space-y-6">
      <Panel
        variant="open"
        title="Pipeline"
        description={`${steps.length} ${steps.length === 1 ? "step" : "steps"}`}
        bodyClassName="p-0"
      >
        {isLoading ? (
          <div className="py-4">
            <Skeleton className="h-16 w-full" />
          </div>
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-4 py-4">
              <div className="flex min-w-0 basis-full items-center gap-2 overflow-x-auto pb-1 sm:flex-1 sm:basis-auto">
                {steps.length === 0 && (
                  <p className="text-sm text-muted-foreground">
                    Add a step to see what it would do to the first {sampleSize}{" "}
                    documents.
                  </p>
                )}

                {steps.map((step, index) => {
                  const processor = processors.find(
                    (p) => p.name === step.processor
                  )
                  if (!processor) return null
                  return (
                    <React.Fragment key={step.processor}>
                      {index > 0 && (
                        <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" />
                      )}
                      <StepChip
                        index={index}
                        processor={processor}
                        isOpen={opened === step.processor}
                        onOpen={() =>
                          setOpened((c) =>
                            c === step.processor ? null : step.processor
                          )
                        }
                        onRemove={() => {
                          remove(index)
                          if (opened === step.processor) setOpened(null)
                        }}
                      />
                    </React.Fragment>
                  )
                })}

                {unused.length > 0 && (
                  <>
                    {steps.length > 0 && (
                      <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" />
                    )}
                    <Select value="" onValueChange={(v) => v && add(v)}>
                      <SelectTrigger
                        aria-label="Add a step"
                        className="h-9 w-auto shrink-0 border-dashed"
                      >
                        <span className="flex items-center gap-1.5 text-muted-foreground">
                          <Plus className="size-3.5" />
                          Add a step
                        </span>
                      </SelectTrigger>
                      <SelectContent>
                        {unused.map((p) => (
                          <SelectItem key={p.name} value={p.name}>
                            {p.title}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </>
                )}
              </div>

              <Button
                className="shrink-0"
                disabled={steps.length === 0 || dataset.status !== "ready"}
                onClick={() => setNaming(true)}
              >
                Run over all {dataset.document_count.toLocaleString()}
              </Button>
            </div>

            {openProcessor && openStep && (
              <StepSettings
                processor={openProcessor}
                step={openStep}
                onChange={(key, value) => update(openIndex, key, value)}
                onSourceSaved={sourceChanged}
              />
            )}
          </>
        )}
      </Panel>

      <PreviewPane
        preview={steps.length === 0 ? null : preview}
        error={steps.length === 0 ? null : previewError}
        steps={steps.length}
      />
      <NameDialog
        key={naming ? "open" : "closed"}
        open={naming}
        source={dataset}
        steps={steps}
        onClose={() => setNaming(false)}
        onQueued={(id) => {
          setNaming(false)
          toast.success("Processing queued.")
          router.push(`/data/${id}`)
        }}
      />
    </div>
  )
}

export interface StepChipProps {
  index: number
  processor: Processor
  isOpen: boolean
  onOpen: () => void
  onRemove: () => void
}

/** One stop on the rail: what it is, and a way to open or drop it. */
function StepChip({
  index,
  processor,
  isOpen,
  onOpen,
  onRemove,
}: StepChipProps) {
  const tunable = processor.params.length > 0
  return (
    <div
      className={cn(
        "flex shrink-0 items-center gap-2 border-y px-3 py-1.5 transition-colors",
        isOpen ? "border-primary/50" : "border-border"
      )}
    >
      <span className="font-mono text-xs text-muted-foreground">
        {index + 1}
      </span>
      <button
        type="button"
        onClick={onOpen}
        title={processor.description}
        aria-expanded={isOpen}
        className="cursor-pointer text-sm font-medium whitespace-nowrap"
      >
        {processor.title}
        {tunable && (
          <ChevronDown
            className={cn(
              "ml-1 inline size-3 text-muted-foreground transition-transform",
              isOpen && "rotate-180"
            )}
          />
        )}
      </button>
      <Button
        variant="ghost"
        size="icon-xs"
        aria-label={`Remove ${processor.title}`}
        onClick={onRemove}
      >
        <X className="size-3" />
      </Button>
    </div>
  )
}

export interface StepSettingsProps {
  processor: Processor
  step: PipelineStep
  onChange: (key: string, value: ParamValue) => void
  onSourceSaved: () => void
}

/** The open step's description and parameters, under the rail it belongs to. */
function StepSettings({
  processor,
  step,
  onChange,
  onSourceSaved,
}: StepSettingsProps) {
  return (
    <div className="border-t border-border py-4">
      <p className="text-xs text-muted-foreground">{processor.description}</p>

      {processor.params.length > 0 && (
        <div className="mt-3 flex flex-wrap items-end gap-x-6 gap-y-3">
          {processor.params.map((param) => {
            const value = step.params?.[param.name] ?? param.default
            const id = `${processor.name}-${param.name}`
            return param.type === "bool" ? (
              <label
                key={param.name}
                className="flex items-center gap-2 text-xs"
              >
                <Checkbox
                  id={id}
                  checked={Boolean(value)}
                  onCheckedChange={(checked) =>
                    onChange(param.name, checked === true)
                  }
                />
                {param.label}
              </label>
            ) : (
              <div key={param.name} className="w-40 space-y-1">
                <label htmlFor={id} className="text-xs text-muted-foreground">
                  {param.label}
                </label>
                <Input
                  id={id}
                  type={param.type === "string" ? "text" : "number"}
                  value={String(value)}
                  min={param.minimum ?? undefined}
                  max={param.maximum ?? undefined}
                  onChange={(e) =>
                    onChange(
                      param.name,
                      param.type === "string"
                        ? e.target.value
                        : Number(e.target.value)
                    )
                  }
                />
              </div>
            )
          })}
        </div>
      )}
      <div className="mt-4 border-t border-border pt-4">
        <ProcessorSource
          key={processor.name}
          name={processor.name}
          onSaved={onSourceSaved}
          actions={
            <Button
              variant="text"
              size="sm"
              nativeButton={false}
              render={
                <Link
                  href={`/processors/${encodeURIComponent(processor.name)}`}
                />
              }
            >
              Open processor
            </Button>
          }
        />
      </div>
    </div>
  )
}

export interface PreviewPaneProps {
  preview: Preview | null
  error: string | null
  steps: number
}

/** Presents aggregate pipeline changes and the processor responsible for each drop. */
function PreviewPane({ preview, error, steps }: PreviewPaneProps) {
  if (error) {
    return (
      <Panel variant="open" title="Before and after">
        <FormError>{error}</FormError>
      </Panel>
    )
  }

  if (steps === 0 || !preview) {
    return (
      <Panel variant="open" title="Before and after">
        <p className="py-10 text-center text-sm text-muted-foreground">
          {steps === 0 ? "Nothing to show yet." : "Working…"}
        </p>
      </Panel>
    )
  }

  return (
    <Panel
      variant="open"
      title="Before and after"
      description={`${preview.sampled} sampled · ${preview.dropped} dropped`}
      bodyClassName="p-0"
    >
      <ul className="divide-y divide-border">
        {preview.results.map((result) => (
          <li key={result.ordinal} className="py-4">
            <div className="mb-2 flex items-center gap-2">
              <span className="text-xs text-muted-foreground">
                #{result.ordinal}
              </span>
              {result.dropped ? (
                <span className="text-xs text-destructive">
                  dropped by {result.dropped_by}
                </span>
              ) : result.changed ? (
                <span className="text-xs text-muted-foreground">changed</span>
              ) : (
                <span className="text-xs text-muted-foreground">unchanged</span>
              )}
            </div>

            <DocumentDiff
              before={documentText(result.before)}
              after={result.dropped ? "" : documentText(result.after)}
              dropped={result.dropped}
            />
          </li>
        ))}
      </ul>
    </Panel>
  )
}

/**
 * One document's before and after, side by side and line by line.
 *
 * Split rather than unified: the two halves are read as two versions of the
 * same text, and a cleaning step's effect is usually a line rewritten in place
 * rather than a line removed and another added.
 *
 * Untouched stretches fold away. A document is long, a step touches little of
 * it, and rendering the rest in full hides the answer among the evidence.
 */
function DocumentDiff({
  before,
  after,
  dropped,
}: {
  before: string
  after: string
  dropped: boolean
}) {
  const grouped = React.useMemo(
    () => (dropped ? [] : hunks(splitDiff(before, after))),
    [before, after, dropped]
  )

  if (dropped) {
    return (
      <div className="grid overflow-hidden border-y border-border sm:grid-cols-2">
        <Side text={before} tone="removed" />
        <div className="flex items-center justify-center border-t border-border px-3 py-6 text-xs text-muted-foreground italic sm:border-t-0 sm:border-l">
          — removed —
        </div>
      </div>
    )
  }

  return (
    <div className="max-h-80 overflow-auto border-y border-border font-mono text-xs">
      <table className="w-full table-fixed border-collapse">
        <colgroup>
          <col className="w-10" />
          <col className="w-[calc(50%-2.5rem)]" />
          <col className="w-10" />
          <col className="w-[calc(50%-2.5rem)]" />
        </colgroup>
        <tbody>
          {grouped.map((hunk, index) =>
            hunk.collapsible ? (
              <Folded key={index} count={hunk.rows.length} />
            ) : (
              hunk.rows.map((row, i) => (
                <Line key={`${index}-${i}`} row={row} />
              ))
            )
          )}
        </tbody>
      </table>
    </div>
  )
}

/** Backgrounds for the two sides of a change, keyed off the chart palette so a
    diff is in the same colours as everything else here rather than GitHub's. */
const TONE = {
  removed:
    "bg-[color-mix(in_oklab,var(--destructive)_13%,transparent)] text-foreground",
  added:
    "bg-[color-mix(in_oklab,var(--chart-2)_16%,transparent)] text-foreground",
  none: "",
} as const

export interface LineProps {
  row: DiffRow
}

/** Aligns one pair of diff lines using the existing added/removed palette. */
function Line({ row }: LineProps) {
  const left =
    row.kind === "removed" || row.kind === "changed" ? "removed" : "none"
  const right =
    row.kind === "added" || row.kind === "changed" ? "added" : "none"

  return (
    <tr className="align-top">
      <Gutter number={row.left?.number} tone={left} />
      <Cell text={row.left?.text} tone={left} />
      <Gutter number={row.right?.number} tone={right} />
      <Cell text={row.right?.text} tone={right} />
    </tr>
  )
}

export interface GutterProps {
  number?: number
  tone: keyof typeof TONE
}

/** Renders a line number with the corresponding change tone. */
function Gutter({ number, tone }: GutterProps) {
  return (
    <td
      className={cn(
        "border-r border-border px-2 py-0.5 text-right align-top text-muted-foreground select-none",
        TONE[tone]
      )}
    >
      {number ?? ""}
    </td>
  )
}

export interface CellProps {
  text?: string
  tone: keyof typeof TONE
}

/** Renders one diff line as literal text with its change tone. */
function Cell({ text, tone }: CellProps) {
  return (
    <td
      className={cn(
        "px-2 py-0.5 wrap-anywhere whitespace-pre-wrap",
        TONE[tone]
      )}
    >
      {text ?? ""}
    </td>
  )
}

export interface FoldedProps {
  count: number
}

/** A run of untouched lines, stated rather than shown. */
function Folded({ count }: FoldedProps) {
  return (
    <tr>
      <td
        colSpan={4}
        className="border-y border-border bg-[color-mix(in_oklab,var(--foreground)_3.5%,transparent)] px-3 py-1 text-center text-2xs text-muted-foreground"
      >
        {count} unchanged {count === 1 ? "line" : "lines"}
      </td>
    </tr>
  )
}

export interface SideProps {
  text: string
  tone: keyof typeof TONE
}

/** The whole of one side, for a document that has no other side to align to. */
function Side({ text, tone }: SideProps) {
  return (
    <pre
      className={cn(
        "max-h-40 overflow-auto px-3 py-2 font-mono text-xs whitespace-pre-wrap",
        TONE[tone]
      )}
    >
      {text}
    </pre>
  )
}

export interface NameDialogProps {
  open: boolean
  source: Dataset
  steps: PipelineStep[]
  onClose: () => void
  onQueued: (datasetId: string) => void
}

/** Collects a derived dataset name and queues the chosen pipeline, retaining failures for retry. */
function NameDialog({
  open,
  source,
  steps,
  onClose,
  onQueued,
}: NameDialogProps) {
  const [name, setName] = React.useState(`${source.name}-clean`)
  const [submitting, setSubmitting] = React.useState(false)
  const [failure, setFailure] = React.useState<string | null>(null)

  /** Queues processing under the entered name and hands the resulting dataset to the caller. */
  async function submit() {
    setSubmitting(true)
    setFailure(null)
    try {
      const queued = await processDataset(source.id, name.trim(), steps)
      onQueued(queued.dataset_id)
    } catch (e) {
      setFailure(e instanceof Error ? e.message : "the run was refused")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Name the result</DialogTitle>
          <DialogDescription>
            A new dataset, with {source.name} left exactly as it is. The steps
            are recorded on it, so what produced it stays answerable.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-1.5">
          <label
            htmlFor="derived-name"
            className="eyebrow text-muted-foreground"
          >
            Call it
          </label>
          <Input
            id="derived-name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>

        {failure && <FormError>{failure}</FormError>}

        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            onClick={submit}
            disabled={submitting || name.trim().length === 0}
          >
            {submitting ? "Queueing…" : "Run it"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
