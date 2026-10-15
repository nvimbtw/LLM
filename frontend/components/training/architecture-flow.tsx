"use client"

import { useState } from "react"
import { Pause, Play } from "lucide-react"

import { Button } from "@/components/ui/button"
import { usePrefersReducedMotion } from "@/hooks/use-reduced-motion"
import { useArchitecturePlayback } from "@/hooks/use-architecture-playback"
import {
  architectureFrame,
  type ArchitectureView,
} from "@/lib/architecture-flow"
import type { TrainConfig } from "@/lib/training"
import { ArchitectureDiagram } from "./architecture-diagram"

export interface ArchitectureFlowProps {
  cfg: Pick<
    TrainConfig,
    "dimensions" | "ctx_window" | "heads" | "layers" | "ff_multiplier"
  > & {
    mode: TrainConfig["mode"] | "rl"
  }
  view: ArchitectureView
}

/** Illustrates training on the composer and generation on model details. One owned clock drives both responsive layouts; pause retains the current transformation and reduced motion shows the complete static schematic. */
export function ArchitectureFlow({ cfg, view }: ArchitectureFlowProps) {
  const reducedMotion = usePrefersReducedMotion()
  const [paused, setPaused] = useState(false)
  const { ref, time } = useArchitecturePlayback(!paused && !reducedMotion)
  const frame = reducedMotion ? null : architectureFrame(view, time)
  return (
    <figure
      ref={ref}
      className="@container mt-6"
      aria-label="Model architecture"
      data-architecture-view={view}
    >
      <figcaption className="mb-5 flex items-center justify-between gap-3 text-xs text-muted-foreground">
        <span>
          {view === "generation" ? "Generation" : "Training"}
          <span className="mx-2 text-border">/</span>Illustrated flow
        </span>
        {!reducedMotion && (
          <Button
            type="button"
            variant="text"
            size="xs"
            className="text-muted-foreground"
            aria-label={
              paused
                ? "Play architecture animation"
                : "Pause architecture animation"
            }
            onClick={() => setPaused(!paused)}
          >
            {paused ? <Play /> : <Pause />}
            {paused ? "Play" : "Pause"}
          </Button>
        )}
      </figcaption>
      <ArchitectureDiagram
        cfg={cfg}
        view={view}
        frame={frame}
        vertical={false}
      />
      <ArchitectureDiagram cfg={cfg} view={view} frame={frame} vertical />
    </figure>
  )
}
