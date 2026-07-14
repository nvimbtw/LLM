"use client"

import * as React from "react"

import {
  createPreset,
  deletePreset,
  listPresets,
  updatePreset,
  type Preset,
} from "@/lib/training-presets"
import type { TrainConfig } from "@/lib/training-socket"

// Owns the user's preset list and the CRUD callbacks. Components consume the list
// + callbacks; they don't fetch themselves. Mutations refetch the list so the
// picker stays consistent (the list is small — a preset set, not a feed).
export function useTrainingPresets() {
  const [presets, setPresets] = React.useState<Preset[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  const refresh = React.useCallback(async () => {
    try {
      setPresets(await listPresets())
    } catch {
      // A failed list is non-fatal — the picker just shows empty.
    } finally {
      setIsLoading(false)
    }
  }, [])

  React.useEffect(() => {
    void refresh()
  }, [refresh])

  // Save the current config. If a preset with this name already exists, update it
  // in place (so "save over" is one action); otherwise create a new one.
  const save = React.useCallback(
    async (name: string, config: TrainConfig) => {
      setError(null)
      try {
        const existing = presets.find((p) => p.name === name)
        if (existing) {
          await updatePreset(existing.id_preset, { name, config })
        } else {
          await createPreset(name, config)
        }
        await refresh()
        return true
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to save preset.")
        return false
      }
    },
    [presets, refresh]
  )

  const remove = React.useCallback(
    async (id: string) => {
      setError(null)
      try {
        await deletePreset(id)
        await refresh()
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to delete preset.")
      }
    },
    [refresh]
  )

  return { presets, isLoading, error, save, remove }
}
