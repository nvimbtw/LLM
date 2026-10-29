"use client"

import * as React from "react"
import { useSessionToken } from "@/hooks/use-auth"

import {
  getDataset,
  listDatasets,
  listDocuments,
  listProcessors,
  listRuns,
  type Dataset,
  type DatasetDocument,
  type DatasetKind,
  type DataRun,
  type Processor,
} from "@/lib/data"

// Loaders for the data page. `useEffect` + `useState` with a numeric
// `refreshKey`, matching every other hook here — there is no query cache in this
// app and adding one for four endpoints would be a second way to fetch.

/** How often the list refreshes while something is still arriving. */
const PULLING_POLL_MS = 2000

interface Loaded<T> {
  data: T
  isLoading: boolean
  error: string | null
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : "something went wrong"
}

/**
 * Every dataset, refreshing on its own only while one is mid-pull.
 *
 * Polling unconditionally would put a request every two seconds behind a page
 * that is usually static; not polling at all would leave a pull looking stuck
 * until the user reloaded. The status of the rows decides which it is.
 */
export function useStoredDatasets(refreshKey = 0): Loaded<Dataset[]> & {
  reload: () => void
} {
  const [state, setState] = React.useState<Loaded<Dataset[]>>({
    data: [],
    isLoading: true,
    error: null,
  })
  const [tick, setTick] = React.useState(0)

  React.useEffect(() => {
    let active = true
    listDatasets()
      .then(
        (data) => active && setState({ data, isLoading: false, error: null })
      )
      .catch(
        (e) =>
          active &&
          setState((s) => ({ ...s, isLoading: false, error: message(e) }))
      )
    return () => {
      active = false
    }
  }, [refreshKey, tick])

  const busy = state.data.some((d) => d.status === "pulling")
  React.useEffect(() => {
    if (!busy) return
    const timer = setInterval(() => setTick((t) => t + 1), PULLING_POLL_MS)
    return () => clearInterval(timer)
  }, [busy])

  return { ...state, reload: () => setTick((t) => t + 1) }
}

export function useDataset(id: string, refreshKey = 0): Loaded<Dataset | null> {
  // The id a result belongs to is stored with it, so navigating to another
  // dataset shows its own loading frame instead of the previous one's content.
  // A refresh of the same id keeps the current content visible while it reloads.
  const [loaded, setLoaded] = React.useState<{
    key: string
    data: Dataset | null
    error: string | null
  } | null>(null)

  React.useEffect(() => {
    let active = true
    getDataset(id)
      .then((data) => {
        if (active) setLoaded({ key: id, data, error: null })
      })
      .catch((e) => {
        if (active) setLoaded({ key: id, data: null, error: message(e) })
      })
    return () => {
      active = false
    }
  }, [id, refreshKey])

  const fresh = loaded !== null && loaded.key === id
  return {
    data: fresh ? loaded.data : null,
    isLoading: !fresh,
    error: fresh ? loaded.error : null,
  }
}

export function useDocuments(
  id: string,
  offset: number,
  limit: number
): Loaded<DatasetDocument[]> & { total: number } {
  // The page a result belongs to is stored *with* it, so "still loading" and
  // "showing the previous page" are the same question, answered by comparison
  // rather than by a second state an effect has to set synchronously.
  const [loaded, setLoaded] = React.useState<{
    key: string
    documents: DatasetDocument[]
    total: number
    error: string | null
  } | null>(null)

  const key = `${id}:${offset}:${limit}`

  React.useEffect(() => {
    let active = true
    listDocuments(id, offset, limit)
      .then((body) => {
        if (!active) return
        setLoaded({
          key,
          documents: body.documents,
          total: body.total,
          error: null,
        })
      })
      .catch((e) => {
        if (active)
          setLoaded({ key, documents: [], total: 0, error: message(e) })
      })
    return () => {
      active = false
    }
  }, [id, offset, limit, key])

  const fresh = loaded !== null && loaded.key === key
  return {
    data: fresh ? loaded.documents : [],
    total: fresh ? loaded.total : 0,
    isLoading: !fresh,
    error: fresh ? loaded.error : null,
  }
}

/** Loads current-session processors, with null kind selecting the full catalogue. */
export function useProcessors(
  kind: DatasetKind | null,
  refreshKey = 0
): Loaded<Processor[]> {
  const token = useSessionToken()
  const [state, setState] = React.useState<Loaded<Processor[]>>({
    data: [],
    isLoading: true,
    error: null,
  })

  React.useEffect(() => {
    if (token === null) return
    let active = true
    listProcessors(kind)
      .then(
        (data) => active && setState({ data, isLoading: false, error: null })
      )
      .catch(
        (e) =>
          active &&
          setState((s) => ({ ...s, isLoading: false, error: message(e) }))
      )
    return () => {
      active = false
    }
  }, [kind, token, refreshKey])

  return state
}

export function useDataRuns(refreshKey = 0): Loaded<DataRun[]> {
  const [state, setState] = React.useState<Loaded<DataRun[]>>({
    data: [],
    isLoading: true,
    error: null,
  })
  const [tick, setTick] = React.useState(0)

  React.useEffect(() => {
    let active = true
    listRuns()
      .then(
        (data) => active && setState({ data, isLoading: false, error: null })
      )
      .catch(
        (e) =>
          active &&
          setState((s) => ({ ...s, isLoading: false, error: message(e) }))
      )
    return () => {
      active = false
    }
  }, [refreshKey, tick])

  const busy = state.data.some(
    (r) => r.status === "running" || r.status === "queued"
  )
  React.useEffect(() => {
    if (!busy) return
    const timer = setInterval(() => setTick((t) => t + 1), PULLING_POLL_MS)
    return () => clearInterval(timer)
  }, [busy])

  return state
}
