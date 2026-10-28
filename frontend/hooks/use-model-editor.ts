"use client"

import { useEffect, useRef, useState } from "react"
import type { ModelDetail, ModelEdits } from "@/hooks/use-model"
import { toast } from "@/components/ui/toast"
import { usePageDraft } from "./use-page-draft"
import { readDraftStrings } from "@/lib/page-draft"

/** Restores text and the explicit supported size category from tab-local metadata without coercing unknown values. */
function modelDraft(value: unknown): ModelEdits {
  const text = readDraftStrings(
    ["display_name", "description", "family"],
    value
  )
  const size =
    value && typeof value === "object" && "size" in value
      ? value.size
      : undefined
  if (
    size !== null &&
    size !== "fast" &&
    size !== "balanced" &&
    size !== "flagship"
  )
    throw new Error("The saved model size is invalid.")
  return { ...text, size }
}

/** Returns editable display metadata, preserving the stored description independently of calculated fallbacks. */
function fieldsOf(model: ModelDetail): ModelEdits {
  return {
    display_name: model.name,
    description: model.storedDescription,
    family: model.family,
    size: model.tier,
  }
}

/** Owns one model's explicit metadata draft and save lifetime. Failed saves retain edits; unmounting ignores late results. */
export function useModelEditor(
  model: ModelDetail,
  save: (fields: ModelEdits) => Promise<void>
) {
  const initial = fieldsOf(model)
  const source = JSON.stringify(initial)
  const draft = usePageDraft(`model:${model.id}`, initial, modelDraft)
  const fields = draft.fields
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const active = useRef(true)
  const pending = useRef(false)
  useEffect(() => {
    active.current = true
    return () => {
      active.current = false
    }
  }, [])

  /** Changes a single display field and clears a previous submission failure. */
  function change<K extends keyof ModelEdits>(key: K, value: ModelEdits[K]) {
    draft.change({ ...fields, [key]: value })
    setError(null)
  }

  /** Persists a validated draft; validation and server failures remain beside the editable page fields. */
  async function submit() {
    if (pending.current) return
    const values = {
      ...fields,
      display_name: fields.display_name.trim(),
      description: fields.description.trim(),
      family: fields.family.trim(),
    }
    if (!values.display_name || !values.family) {
      setError("Name and family can't be empty.")
      return
    }
    if (
      [values.display_name, values.family].some(
        (value) => Array.from(value).length > 120
      )
    ) {
      setError("Name and family must be at most 120 characters.")
      return
    }
    pending.current = true
    setSaving(true)
    setError(null)
    try {
      await save(values)
      if (active.current) {
        draft.change(values)
        draft.forget()
        toast.success("Model updated.", values.display_name)
      }
    } catch (failure) {
      if (active.current)
        setError(
          failure instanceof Error ? failure.message : "Couldn't save changes."
        )
    } finally {
      pending.current = false
      if (active.current) setSaving(false)
    }
  }

  /** Discards unsaved fields and restores the current persisted metadata. */
  function reset() {
    if (pending.current) return
    draft.reset()
    setError(null)
  }

  return {
    fields,
    change,
    submit,
    reset,
    saving,
    error: error ?? draft.persistenceError,
    hasChanges: JSON.stringify(fields) !== source,
  }
}
