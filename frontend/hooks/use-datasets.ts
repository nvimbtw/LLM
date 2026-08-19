"use client"

import * as React from "react"

import { listDatasets } from "@/lib/data"
import type { DatasetInfo } from "@/lib/training"

// Loads the selectable datasets for the params rail's checkbox picker: the
// prose corpora, or the conversation sets when the run is a fine-tune.
//
// One catalogue, filtered by kind. Both used to be folders under the engine's
// data root and were two different routes; they are now rows in the data
// engine's database, told apart by the column that already tells them apart.
// A dataset that is still pulling is left out — a run that selected one would
// pack whatever had landed by the time it reached the front of the queue, and
// key that as though it were the whole thing.
//
// One hook rather than two because the rail renders one picker and only its
// source changes — two hooks would mean both fetching on every render of a page
// that can only use one of them.
export function useDatasets(sft = false) {
  // The mode each result belongs to is stored *with* it, so "still loading" and
  // "showing the other mode's list" are the same question and are answered by
  // comparison rather than by a second state the effect has to reset. Serving a
  // stale list across a mode flip would offer prose corpora to a fine-tune.
  const [loaded, setLoaded] = React.useState<{
    sft: boolean
    datasets: DatasetInfo[]
  } | null>(null)

  React.useEffect(() => {
    let active = true
    listDatasets()
      .then((rows) =>
        rows
          .filter((d) => d.kind === (sft ? "sft" : "pretrain") && d.status === "ready")
          .map((d) => ({ id: d.id, name: d.name, documents: d.document_count }))
      )
      // A rejected fetch still settles, or the picker would sit on "Loading…"
      // forever; the empty list renders the "nothing pulled yet" hint instead.
      .catch(() => [] as DatasetInfo[])
      .then((datasets) => {
        if (active) setLoaded({ sft, datasets })
      })
    return () => {
      active = false
    }
  }, [sft])

  const fresh = loaded !== null && loaded.sft === sft
  return {
    datasets: fresh ? loaded.datasets : [],
    isLoading: !fresh,
  }
}
