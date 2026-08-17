"use client"

import * as React from "react"

import { listDatasets, listSftDatasets, type DatasetInfo } from "@/lib/training"

// Loads the selectable datasets for the params rail's checkbox picker: the
// pretrain corpora (subfolders of uploads/, each with its .txt count), or the
// SFT conversation sets when the run is a fine-tune. Fetched on mount and
// whenever the mode flips; either set only changes when files are added on
// disk, so no live refresh is needed.
//
// One hook rather than two because the rail renders one picker and only its
// source changes — two hooks would mean both fetching on every render of a page
// that can only use one of them.
export function useDatasets(sft = false) {
  // The mode each result belongs to is stored *with* it, so "still loading" and
  // "showing the other mode's list" are the same question and are answered by
  // comparison rather than by a second state the effect has to reset. The two
  // catalogues share no names, so serving a stale one across a mode flip would
  // offer prose corpora to a fine-tune.
  const [loaded, setLoaded] = React.useState<{
    sft: boolean
    datasets: DatasetInfo[]
  } | null>(null)

  React.useEffect(() => {
    let active = true
    const load = sft ? listSftDatasets : listDatasets
    load()
      // A rejected fetch still settles, or the picker would sit on "Loading…"
      // forever; the empty list renders the "nothing on disk" hint instead.
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
