"use client"

import * as React from "react"

import { listDatasets, type DatasetInfo } from "@/lib/training"

// Loads the selectable pretrain datasets (subfolders of uploads/, each with its
// .txt count) for the params rail's checkbox picker. Fetched once on mount; the
// set only changes when files are added on disk, so no live refresh is needed.
export function useDatasets() {
  const [datasets, setDatasets] = React.useState<DatasetInfo[]>([])
  const [isLoading, setIsLoading] = React.useState(true)

  React.useEffect(() => {
    let active = true
    listDatasets()
      .then((d) => {
        if (active) setDatasets(d)
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
    }
  }, [])

  return { datasets, isLoading }
}
