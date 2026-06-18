"use client"

import { useModels } from "@/hooks/use-models"
import { Skeleton } from "@/components/ui/skeleton"
import { ModelCard } from "./model-card"

export function ModelsGrid() {
  const { models, isLoading, error } = useModels()

  if (error) {
    return (
      <p className="text-sm text-destructive">
        Failed to load models. Please try again.
      </p>
    )
  }

  if (isLoading) {
    return (
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-64 rounded-xl" />
        ))}
      </div>
    )
  }

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {models.map((model) => (
        <ModelCard key={model.id} model={model} />
      ))}
    </div>
  )
}
