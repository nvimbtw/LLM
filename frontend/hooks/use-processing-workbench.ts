"use client"

import * as React from "react"

import { useProcessors } from "@/hooks/use-data"
import {
  preview as runPreview,
  type Dataset,
  type ParamValue,
  type PipelineStep,
  type Preview,
} from "@/lib/data"

const PREVIEW_DEBOUNCE_MS = 250
const PREVIEW_SAMPLE = 8

interface PreviewState {
  key: string
  preview: Preview | null
  error: string | null
}

/** Owns one keyed workbench's pipeline, catalogue refreshes and cancellable previews. */
export function useProcessingWorkbench(dataset: Dataset) {
  const [revision, setRevision] = React.useState(0)
  const {
    data: processors,
    isLoading,
    error,
  } = useProcessors(dataset.kind, revision)
  const [steps, setSteps] = React.useState<PipelineStep[]>([])
  const [result, setResult] = React.useState<PreviewState | null>(null)
  const key = JSON.stringify([dataset.id, steps, revision])

  React.useEffect(() => {
    if (steps.length === 0) return
    const controller = new AbortController()
    const timer = setTimeout(() => {
      runPreview(dataset.id, steps, PREVIEW_SAMPLE, controller.signal)
        .then((preview) => {
          if (!controller.signal.aborted)
            setResult({ key, preview, error: null })
        })
        .catch((failure: unknown) => {
          if (!controller.signal.aborted)
            setResult({
              key,
              preview: null,
              error:
                failure instanceof Error ? failure.message : "Preview failed",
            })
        })
    }, PREVIEW_DEBOUNCE_MS)
    return () => {
      clearTimeout(timer)
      controller.abort()
    }
  }, [dataset.id, key, steps])

  /** Appends a catalogue step with its declared defaults within the API's step budget. */
  function add(name: string) {
    const processor = processors.find((item) => item.name === name)
    if (!processor) return
    const params = Object.fromEntries(
      processor.params.map((param) => [param.name, param.default])
    )
    setSteps((current) =>
      current.length >= 32 ? current : [...current, { processor: name, params }]
    )
  }

  /** Removes a pipeline step without changing the remaining step parameters. */
  function remove(index: number) {
    setSteps((current) => current.filter((_, i) => i !== index))
  }

  /** Replaces one parameter locally; the debounced preview applies declared coercion. */
  function update(index: number, name: string, value: ParamValue) {
    setSteps((current) =>
      current.map((step, i) =>
        i === index
          ? { ...step, params: { ...step.params, [name]: value } }
          : step
      )
    )
  }

  /** Reloads catalogue metadata and previews after either a valid or failed source reload. */
  function sourceChanged() {
    setRevision((current) => current + 1)
  }

  return {
    processors,
    isLoading,
    error,
    steps,
    add,
    remove,
    update,
    sourceChanged,
    preview: result?.key === key ? result.preview : null,
    previewError: result?.key === key ? result.error : null,
    unused:
      steps.length >= 32
        ? []
        : processors.filter(
            (item) => !steps.some((step) => step.processor === item.name)
          ),
    sampleSize: PREVIEW_SAMPLE,
  }
}
