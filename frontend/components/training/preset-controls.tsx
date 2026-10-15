"use client"

import * as React from "react"
import { Check, ChevronDown, Trash2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { useModels } from "@/hooks/use-models"
import { useTrainingPresets } from "@/hooks/use-training-presets"
import { loadModelConfig, toTrainConfig } from "@/lib/training-presets"
import type { TrainConfig } from "@/lib/training"
import { FormError } from "@/components/ui/error-state"

export interface PresetControlsProps {
  currentConfig: TrainConfig
  disabled: boolean
  onLoad: (cfg: TrainConfig) => void
  modelsRefreshKey?: unknown
}

/** Loads saved or model-derived training configs and saves named presets; failed model reads leave the draft intact. */
export function PresetControls({
  currentConfig,
  disabled,
  onLoad,
  modelsRefreshKey,
}: PresetControlsProps) {
  const { presets, save, remove, error } = useTrainingPresets()
  const { models } = useModels(modelsRefreshKey)
  const [saving, setSaving] = React.useState(false)
  const [name, setName] = React.useState("")

  /** Opens preset naming with the current model name as its initial value. */
  const beginSave = () => {
    setName(currentConfig.model_name)
    setSaving(true)
  }

  /** Persists a nonblank preset name and closes naming only after success. */
  const commitSave = async () => {
    const trimmed = name.trim()
    if (!trimmed) return
    const ok = await save(trimmed, currentConfig)
    if (ok) setSaving(false)
  }

  /** Loads the supplied model config into the editor; failed reads preserve its existing draft. */
  const loadModel = async (id: string, modelName: string) => {
    try {
      onLoad(await loadModelConfig(id, modelName))
    } catch {}
  }

  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5">
        <DropdownMenu>
          <DropdownMenuTrigger
            render={
              <Button
                type="button"
                variant="text"
                size="sm"
                disabled={disabled}
                className="flex-1 justify-between font-normal text-muted-foreground"
              >
                Presets
                <ChevronDown className="size-4 opacity-60" />
              </Button>
            }
          />
          <DropdownMenuContent align="start" className="w-56">
            <DropdownMenuGroup>
              <DropdownMenuLabel>Saved presets</DropdownMenuLabel>
              {presets.length === 0 ? (
                <div className="px-2 py-1.5 text-xs text-muted-foreground">
                  No saved presets yet.
                </div>
              ) : (
                presets.map((p) => (
                  <DropdownMenuItem
                    key={p.id_preset}
                    onClick={() =>
                      onLoad(
                        toTrainConfig(
                          p.config as unknown as Record<string, unknown>,
                          p.config?.model_name ?? "my-model"
                        )
                      )
                    }
                    className="justify-between"
                  >
                    <span className="truncate">{p.name}</span>
                    <button
                      type="button"
                      aria-label={`Delete ${p.name}`}
                      onClick={(e) => {
                        e.stopPropagation()
                        e.preventDefault()
                        void remove(p.id_preset)
                      }}
                      className="ml-2 text-muted-foreground hover:text-destructive"
                    >
                      <Trash2 className="size-3.5" />
                    </button>
                  </DropdownMenuItem>
                ))
              )}
            </DropdownMenuGroup>
            <DropdownMenuSeparator />
            <DropdownMenuSub>
              <DropdownMenuSubTrigger>Load from model</DropdownMenuSubTrigger>
              <DropdownMenuSubContent className="max-h-64 overflow-y-auto">
                {models.length === 0 ? (
                  <div className="px-2 py-1.5 text-xs text-muted-foreground">
                    No models yet.
                  </div>
                ) : (
                  models.map((m) => (
                    <DropdownMenuItem
                      key={m.id}
                      onClick={() => void loadModel(m.id, m.name)}
                    >
                      <span className="truncate">{m.name}</span>
                    </DropdownMenuItem>
                  ))
                )}
              </DropdownMenuSubContent>
            </DropdownMenuSub>
          </DropdownMenuContent>
        </DropdownMenu>

        {!saving && (
          <Button
            type="button"
            variant="text"
            size="sm"
            disabled={disabled}
            onClick={beginSave}
          >
            Save
          </Button>
        )}
      </div>

      {saving && (
        <div className="flex items-center gap-1.5">
          <Input
            variant="inline"
            aria-label="Preset name"
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault()
                void commitSave()
              } else if (e.key === "Escape") {
                setSaving(false)
              }
            }}
            placeholder="Preset name"
            className="h-8"
          />
          <Button
            type="button"
            size="icon"
            variant="text"
            className="size-8 shrink-0"
            aria-label="Confirm save"
            onClick={() => void commitSave()}
          >
            <Check className="size-4" />
          </Button>
        </div>
      )}

      {error && <FormError className="text-xs">{error}</FormError>}
    </div>
  )
}
