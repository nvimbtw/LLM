"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { ArchitectureReadOwner } from "@/lib/architecture/read-owner"
import { architectures } from "@/lib/architectures"
import type { TrainConfig } from "@/lib/training"
import type { TrainingArchitecturePreview } from "@/lib/architecture/contracts.generated"
import {
  trainingPreviewRequest,
  trainingSelectionKey,
} from "@/lib/training-selection"

/** Owns explicit metadata-only profile reads for one actual session/pin; changed form inputs immediately stale old previews and no edit automatically starts numerical work. */
export function useTrainingArchitecturePreview(config: TrainConfig) {
  const token = useSessionToken()
  const revision = config.architecture?.revision_id ?? null
  const resource = `${token}:${revision}`
  const key = trainingSelectionKey(config)
  const reader = React.useRef<{
    resource: string
    owner: ArchitectureReadOwner
  } | null>(null)
  const [state, setState] = React.useState<{
    resource: string
    key: string
    loading: boolean
    value: TrainingArchitecturePreview | null
    error: string | null
  } | null>(null)
  React.useEffect(() => {
    if (!token || !revision) return
    const owner = new ArchitectureReadOwner({ token, resource })
    reader.current = { resource, owner }
    return () => owner.dispose()
  }, [token, revision, resource])

  /** Reads one captured selection/control snapshot, retaining its identity for stale-result presentation and refusing foreign/late replies. */
  async function load(): Promise<void> {
    const current = reader.current
    if (!token || !revision || current?.resource !== resource) return
    setState({ resource, key, value: null, loading: true, error: null })
    try {
      const request = trainingPreviewRequest(config)
      const value = await current.owner.read((signal) =>
        architectures.trainingPreview(request, { token: token!, signal })
      )
      if (value) setState({ resource, key, value, loading: false, error: null })
    } catch (cause) {
      setState({
        resource,
        key,
        value: null,
        loading: false,
        error:
          cause instanceof Error
            ? cause.message.slice(0, 2000)
            : "Training profile did not load.",
      })
    }
  }
  const current = state?.resource === resource ? state : null
  return {
    value: current?.value ?? null,
    error: current?.error ?? null,
    loading: current?.loading ?? false,
    stale: current?.key !== key || !current?.value,
    load,
  }
}
