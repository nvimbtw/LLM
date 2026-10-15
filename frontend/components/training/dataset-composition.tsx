"use client"

import { useMemo } from "react"
import { Pie, PieChart } from "recharts"

import { datasetColor } from "@/components/training/chart-theme"
import { Figure } from "@/components/training/stat-run"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import { datasetComposition } from "@/lib/training-data"
import type { DatasetInfo, TrainConfig } from "@/lib/training"

export interface DatasetCompositionChartProps {
  cfg: TrainConfig
  datasets: DatasetInfo[]
  loading: boolean
}

/** Shows unique selected document counts and shares, updating with limits and curriculum. Unavailable sizes never appear as measured zero or a complete pie. */
export function DatasetCompositionChart({
  cfg,
  datasets,
  loading,
}: DatasetCompositionChartProps) {
  const composition = useMemo(
    () => datasetComposition(cfg, datasets),
    [cfg, datasets]
  )
  const slices = useMemo(
    () =>
      composition.datasets
        .filter((dataset) => dataset.documents > 0)
        .map((dataset) => ({
          ...dataset,
          fill: datasetColor(dataset.catalogueIndex),
        })),
    [composition]
  )
  if (loading) return null
  if (!composition.complete)
    return (
      <p className="text-xs text-muted-foreground">
        Dataset composition is unavailable until all selected sizes and limits
        are valid.
      </p>
    )
  if (composition.datasets.length === 0) return null
  if (composition.total === 0)
    return (
      <p className="text-xs text-muted-foreground">
        The selected datasets contain no documents.
      </p>
    )

  return (
    <Figure
      caption="Dataset composition"
      aside={`${composition.total.toLocaleString()} documents`}
      minHeight="0"
    >
      <div className="grid items-center gap-6 sm:grid-cols-[11rem_minmax(0,1fr)]">
        <ChartContainer
          config={{ documents: { label: "Documents" } }}
          className="mx-auto aspect-square h-44 w-44"
          aria-label="Selected document shares by dataset"
        >
          <PieChart>
            <Pie
              data={slices}
              dataKey="documents"
              nameKey="name"
              innerRadius={48}
              outerRadius={76}
              stroke="var(--background)"
              strokeWidth={2}
              isAnimationActive={false}
            />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  hideLabel
                  formatter={(value, name) => (
                    <span>
                      {name} · {Number(value).toLocaleString()} documents
                    </span>
                  )}
                />
              }
            />
          </PieChart>
        </ChartContainer>
        <ul className="min-w-0 space-y-3">
          {composition.datasets.map((dataset) => (
            <li
              key={dataset.id}
              className="flex min-w-0 items-center justify-between gap-4 text-xs"
            >
              <span className="flex min-w-0 items-center gap-2">
                <span
                  aria-hidden
                  className="size-2 shrink-0 rounded-full"
                  style={{
                    backgroundColor: datasetColor(dataset.catalogueIndex),
                  }}
                />
                <span className="truncate">{dataset.name}</span>
              </span>
              <span className="shrink-0 text-right tabular-nums">
                {dataset.documents.toLocaleString()}
                <span className="ml-3 inline-block w-12 text-muted-foreground">
                  {((dataset.documents / composition.total) * 100).toFixed(1)}%
                </span>
              </span>
            </li>
          ))}
        </ul>
      </div>
      {Boolean(cfg.curriculum?.length) && (
        <p className="text-xs text-muted-foreground">
          Unique documents across stages; epoch repetitions are excluded.
        </p>
      )}
    </Figure>
  )
}
