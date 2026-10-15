import type { DatasetInfo, DatasetSelection, TrainConfig } from "./training"

export interface DatasetShare {
  id: string
  name: string
  documents: number
  catalogueIndex: number
}

export interface DatasetComposition {
  datasets: DatasetShare[]
  total: number
  complete: boolean
}

/** Resolves selected document counts against the catalogue. Repeated curriculum selections use their largest prefix, counting unique documents once rather than weighting by epochs. Unknown sizes or invalid draft limits make the result incomplete. */
export function datasetComposition(
  cfg: TrainConfig,
  catalogue: DatasetInfo[]
): DatasetComposition {
  const selection: DatasetSelection[] = cfg.curriculum?.length
    ? cfg.curriculum.flatMap((stage) => stage.datasets)
    : (cfg.datasets ?? [])
  const shares = new Map<string, DatasetShare>()
  let complete = true
  for (const selected of selection) {
    const catalogueIndex = catalogue.findIndex(
      (dataset) => dataset.id === selected.id
    )
    const dataset = catalogue[catalogueIndex]
    if (
      !dataset ||
      !Number.isSafeInteger(dataset.documents) ||
      dataset.documents < 0 ||
      !Number.isSafeInteger(selected.count) ||
      selected.count < 0
    ) {
      complete = false
      continue
    }
    const documents =
      selected.count === 0
        ? dataset.documents
        : Math.min(selected.count, dataset.documents)
    const previous = shares.get(dataset.id)
    shares.set(dataset.id, {
      id: dataset.id,
      name: dataset.name,
      documents: Math.max(previous?.documents ?? 0, documents),
      catalogueIndex,
    })
  }
  const datasets = [...shares.values()]
  const total = datasets.reduce((sum, dataset) => sum + dataset.documents, 0)
  return { datasets, total, complete: complete && Number.isSafeInteger(total) }
}
