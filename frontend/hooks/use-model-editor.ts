"use client"

import { useEffect, useRef, useState } from "react"
import type { ModelDetail, ModelEdits } from "@/hooks/use-model"
import { toast } from "@/components/ui/toast"

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
  const [loadedFor, setLoadedFor] = useState(source)
  const [fields, setFields] = useState(initial)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const active = useRef(true)
  if (loadedFor !== source) {
    setLoadedFor(source)
    setFields(initial)
    setError(null)
  }
  useEffect(() => {
    active.current = true
    return () => {
      active.current = false
    }
  }, [])

  /** Changes a single display field and clears a previous submission failure. */
  function change<K extends keyof ModelEdits>(key: K, value: ModelEdits[K]) {
    setFields((previous) => ({ ...previous, [key]: value }))
    setError(null)
  }

  /** Persists a validated draft; validation and server failures remain beside the editable page fields. */
  async function submit() {
    if (saving) return
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
    setSaving(true)
    setError(null)
    try {
      await save(values)
      if (active.current) toast.success("Model updated.", values.display_name)
    } catch (failure) {
      if (active.current)
        setError(
          failure instanceof Error ? failure.message : "Couldn't save changes."
        )
    } finally {
      if (active.current) setSaving(false)
    }
  }

  /** Discards unsaved fields and restores the current persisted metadata. */
  function reset() {
    setFields(initial)
    setError(null)
  }

  return {
    fields,
    change,
    submit,
    reset,
    saving,
    error,
    hasChanges: JSON.stringify(fields) !== source,
  }
}
