"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"

/** How long the id→name map is reused; it changes only when a run finishes or a model is deleted. */
const CACHE_MS = 60_000

let cached: { names: Map<string, string>; at: number } | null = null
let inFlight: Promise<Map<string, string>> | null = null

/**
 * Reads the name map once per cache window and shares a concurrent read.
 *
 * The gym page mounts this hook twice — the run list and the open run — and the
 * list request carries every checkpoint, so asking twice is the expensive half
 * of a page that is otherwise one small query.
 */
function loadModelNames(): Promise<Map<string, string>> {
  if (cached && Date.now() - cached.at < CACHE_MS)
    return Promise.resolve(cached.names)
  if (inFlight) return inFlight

  const promise = apiFetch(`/models?checkpoints=true`)
    .then((res) => {
      if (!res.ok) throw new Error("Unable to load models.")
      return res.json() as Promise<{ models: { id: string; name: string }[] }>
    })
    .then((data) => {
      const names = new Map(data.models.map((m) => [m.id, m.name]))
      cached = { names, at: Date.now() }
      return names
    })
    .finally(() => {
      if (inFlight === promise) inFlight = null
    })

  inFlight = promise
  return promise
}

/** Id → display name, **checkpoints included**.
 *
 * For rendering an id somebody else recorded, which is not the same job as
 * offering a model to pick. `useModels` asks for `/models` bare and must keep
 * doing so — one run's ten intermediates would outnumber every finished model
 * in the grid, the chat picker and the benchmark composer at once — but a gym
 * run *row* is reading back what was measured, and what a gym measures is very
 * often a checkpoint: an RL run produces one per iteration and the whole point
 * of the held-out gym run is to score them.
 *
 * Without this the row falls back to a bare uuid, which is exactly the case a
 * reader most needs named: `Nightly SFT #7` says which run and which iteration,
 * and `4f3c…` says nothing at all.
 *
 * A map rather than a list because every caller looks one id up. */
export function useModelNames(): {
  names: Map<string, string>
  isLoading: boolean
} {
  const [names, setNames] = React.useState<Map<string, string>>(() => new Map())
  const [isLoading, setIsLoading] = React.useState(true)

  React.useEffect(() => {
    let active = true
    loadModelNames()
      .then((loaded) => {
        if (active) setNames(loaded)
      })
      // Named or not named: a row whose model cannot be looked up still renders,
      // with the id it recorded. Failing the list over a label would be worse
      // than the label being a uuid.
      .catch(() => undefined)
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
    }
  }, [])

  return { names, isLoading }
}
