"use client"

import Link from "next/link"

import { LoadError } from "@/components/ui/error-state"
import { Panel } from "@/components/ui/panel"
import { Skeleton } from "@/components/ui/skeleton"
import {
  useModelLineage,
  type LineageNode,
  type LineageRelation,
  type ModelDetail,
} from "@/hooks/use-model"
import type { LineageDatasetRef } from "@/hooks/use-models"
import { formatNumber } from "./format"

export interface ModelLineageProps {
  model: ModelDetail
}

const RELATION_LABEL: Record<LineageRelation, string> = {
  "fine-tuned": "Fine-tuned from",
  continued: "Continued from",
}

/**
 * How the model was constructed: the chain of models it was made from, and —
 * for each one — what it actually trained on.
 *
 * Two kinds of edge make the chain, never a tree: `baseModelId` (a fine-tune's
 * ancestry) and `resumedFromId` (a pretrain continuing from a checkpoint). A
 * model states at most one of the two — they belong to different `family`
 * values — so this is always a list read top to bottom, never a branching
 * graph, and plain markup draws it without reaching for SVG.
 *
 * **A checkpoint is deliberately never a node here.** `parentId`/
 * `checkpointIndex` say "one of this run's own saves", which is `Model
 * Checkpoints`' relationship, not this one — a checkpoint is not a different
 * recipe the model was built from, it is the same recipe mid-run. Folding it
 * in would draw a run's ten intermediates as though they were ten ancestors.
 *
 * Absent entirely when there is nothing to draw — a from-scratch pretrain with
 * no ancestry — the same reasoning `ModelCheckpoints` uses for a run that left
 * no intermediates: an empty panel reads as one that failed to load.
 */
export function ModelLineage({ model }: ModelLineageProps) {
  const { nodes, resolveDatasetName, isLoading, error, retry } =
    useModelLineage(model)

  if (!isLoading && !error && nodes.length <= 1) return null

  return (
    <Panel variant="open" title="Lineage" bodyClassName="p-0">
      {isLoading ? (
        <Skeleton className="m-5 h-28 rounded-(--radius-sheet)" />
      ) : error ? (
        <LoadError title={error.message} onRetry={retry} className="m-5" />
      ) : (
        <ol>
          {nodes.map((node, i) => {
            const isLast = i === nodes.length - 1
            return (
              <li key={node.id}>
                <LineageRow
                  node={node}
                  resolveDatasetName={resolveDatasetName}
                />

                {!isLast && node.relation && (
                  <div className="flex items-center gap-1.5 pb-3 text-2xs text-muted-foreground">
                    <span aria-hidden className="text-muted-foreground/50">
                      ↓
                    </span>
                    {RELATION_LABEL[node.relation]}
                  </div>
                )}
              </li>
            )
          })}
        </ol>
      )}
    </Panel>
  )
}

/** Renders one model and its resolved training datasets, linking ancestor models to their detail pages. */
function LineageRow({
  node,
  resolveDatasetName,
}: {
  node: LineageNode
  resolveDatasetName: (dataset: LineageDatasetRef) => string
}) {
  const trained = node.datasets.map((d) => {
    const name = resolveDatasetName(d)
    return d.count ? `${name} (${formatNumber(d.count)})` : name
  })

  return (
    <div className="pt-3.5 pb-2">
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        {node.isRoot ? (
          <span className="text-sm font-semibold">{node.name}</span>
        ) : (
          <Link
            href={`/models/${node.id}`}
            className="text-sm font-semibold hover:underline"
          >
            {node.name}
          </Link>
        )}
        <span className="text-xs text-muted-foreground">{node.family}</span>
        {node.isRoot && (
          <span className="text-xs text-primary">This model</span>
        )}
      </div>
      <p className="mt-1 text-xs text-muted-foreground">
        {trained.length > 0
          ? `Trained on ${trained.join(", ")}`
          : "No recorded training data"}
      </p>
    </div>
  )
}
