"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { getToken } from "@/lib/api"
import {
  createPreset,
  deletePreset,
  listPresets,
  updatePreset,
  type Preset,
  type PresetRequest,
} from "@/lib/training-presets"
import type { TrainConfig } from "@/lib/training"

interface PresetOwner {
  resource: string
  active: boolean
  loaded: boolean
  writing: boolean
  controllers: Set<AbortController>
}

/** Owns one actual-session preset list and serialized explicit writes; replaced sessions abort transport, mask old settings and ignore late acknowledgements. */
export function useTrainingPresets() {
  const token = useSessionToken()
  const [epoch, retry] = React.useReducer((value: number) => value + 1, 0)
  const resource = `${token}:${epoch}`
  const [state, setState] = React.useState<{
    resource: string
    presets: Preset[]
    error: string | null
  } | null>(null)
  const lifetime = React.useRef<PresetOwner | null>(null)
  React.useEffect(() => {
    if (!token) return
    const owner: PresetOwner = {
      resource,
      active: true,
      loaded: false,
      writing: false,
      controllers: new Set(),
    }
    lifetime.current = owner
    const controller = new AbortController()
    owner.controllers.add(controller)
    void listPresets({ token, signal: controller.signal })
      .then((presets) => {
        if (owner.active) {
          owner.loaded = true
          setState({ resource, presets, error: null })
        }
      })
      .catch((cause: unknown) => {
        if (owner.active)
          setState({
            resource,
            presets: [],
            error:
              cause instanceof Error
                ? cause.message.slice(0, 2000)
                : "Presets did not load.",
          })
      })
      .finally(() => owner.controllers.delete(controller))
    return () => {
      owner.active = false
      owner.controllers.forEach((value) => value.abort())
      owner.controllers.clear()
    }
  }, [token, resource])
  const current = state?.resource === resource ? state : null

  /** Applies one explicit preset mutation only after the current list loaded, then reads authoritative saved envelopes before publishing success. */
  const writePreset = React.useCallback(
    async (
      mutation: (request: PresetRequest) => Promise<unknown>
    ): Promise<boolean> => {
      const owner = lifetime.current
      if (
        !token ||
        !owner?.active ||
        owner.resource !== resource ||
        !owner.loaded ||
        owner.writing ||
        getToken() !== token
      )
        return false
      owner.writing = true
      const controller = new AbortController()
      owner.controllers.add(controller)
      try {
        await mutation({ token, signal: controller.signal })
        const presets = await listPresets({ token, signal: controller.signal })
        if (!owner.active || getToken() !== token) return false
        setState({ resource, presets, error: null })
        return true
      } catch (cause) {
        if (owner.active && getToken() === token)
          setState((previous) =>
            previous?.resource === resource
              ? {
                  ...previous,
                  error:
                    cause instanceof Error
                      ? cause.message.slice(0, 2000)
                      : "Preset operation failed. Refresh to check its saved result.",
                }
              : previous
          )
        return false
      } finally {
        owner.writing = false
        owner.controllers.delete(controller)
      }
    },
    [token, resource]
  )

  /** Saves the explicit name/settings, updating a same-name preset only from this session's currently displayed list. */
  const save = React.useCallback(
    (name: string, config: TrainConfig): Promise<boolean> => {
      const existing = current?.presets.find((preset) => preset.name === name)
      return writePreset((request) =>
        existing
          ? updatePreset(existing.id_preset, { name, config }, request)
          : createPreset(name, config, request)
      )
    },
    [current, writePreset]
  )
  /** Deletes a selected preset under the same serialized owner and refreshes its authoritative list. */
  const remove = React.useCallback(
    (id: string): Promise<boolean> =>
      writePreset((request) => deletePreset(id, request)),
    [writePreset]
  )
  return {
    presets: current?.presets ?? [],
    isLoading: Boolean(token && !current),
    error: current?.error ?? null,
    save,
    remove,
    retry,
  }
}
