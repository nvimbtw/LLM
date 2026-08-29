"use client"

import * as React from "react"
import { Check, ChevronDown } from "lucide-react"

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { InputGroupButton } from "@/components/ui/input-group"
import { CheckpointSelector } from "./checkpoint-selector"
import { cn } from "@/lib/utils"
import { useCheckpointParent } from "@/hooks/use-checkpoint-parent"
import { useModels, type Model } from "@/hooks/use-models"

export function ModelSelector({
  value,
  onValueChange,
}: {
  value: string
  onValueChange: (value: string) => void
}) {
  const { models, isLoading } = useModels()
  // The choice may name a checkpoint, which is deliberately in no model list —
  // so `value` alone cannot say which model is selected, and an id missing from
  // the list is not yet evidence of anything.
  const { parentId, isResolving, record } = useCheckpointParent(value, models)

  // Default to the first model once the list has loaded — and correct a
  // persisted choice that no longer exists (deleted or unpublished since a
  // past session), which otherwise shows a blank trigger forever.
  //
  // Waits on `isResolving`: a checkpoint is absent from `models` exactly as a
  // deleted model is, and correcting before the two are told apart would drop
  // every checkpoint choice the moment it was made.
  React.useEffect(() => {
    if (models.length > 0 && !isResolving && !parentId) {
      onValueChange(models[0].id)
    }
  }, [models, isResolving, parentId, onValueChange])

  const selected = models.find((m) => m.id === parentId)

  // Group models under their family, families sorted alphabetically.
  const groups = React.useMemo(() => {
    const byFamily = new Map<string, Model[]>()
    for (const m of models) {
      const family = m.family || "Custom"
      const list = byFamily.get(family)
      if (list) list.push(m)
      else byFamily.set(family, [m])
    }
    return Array.from(byFamily.entries()).sort(([a], [b]) => a.localeCompare(b))
  }, [models])

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger
          render={
            <InputGroupButton
              aria-label="Select model"
              disabled={isLoading || models.length === 0}
            >
              {selected?.name ?? (isLoading ? "Loading…" : "Select model")}
              <ChevronDown className="text-muted-foreground" />
            </InputGroupButton>
          }
        />
        {/* One flat, scrollable list rather than a family → model flyout — a
          second panel hovering off the first is chrome the choice doesn't
          need. Families are hairline-divided section labels, not their own
          row, so there's nothing here to open except the menu itself. */}
        <DropdownMenuContent
          side="top"
          align="start"
          className="max-h-72 min-w-56 overflow-y-auto p-1"
        >
          {groups.map(([family, familyModels], i) => (
            <React.Fragment key={family}>
              {i > 0 && <DropdownMenuSeparator />}
              <DropdownMenuGroup>
                <DropdownMenuLabel className="eyebrow text-muted-foreground/80">
                  {family}
                </DropdownMenuLabel>
                {familyModels.map((model) => {
                  // Marked against the *model* the choice belongs to, so picking
                  // one of its checkpoints still shows which model you are in.
                  const isActive = model.id === parentId
                  return (
                    <DropdownMenuItem
                      key={model.id}
                      onClick={() => onValueChange(model.id)}
                      className="gap-3 py-1.5"
                    >
                      <Check
                        className={cn(
                          "size-3.5 text-primary transition-opacity",
                          isActive ? "opacity-100" : "opacity-0"
                        )}
                      />
                      <span className={cn(isActive && "font-medium")}>
                        {model.name}
                      </span>
                      <span className="ml-auto eyebrow text-muted-foreground/70">
                        {model.tier}
                      </span>
                    </DropdownMenuItem>
                  )
                })}
              </DropdownMenuGroup>
            </React.Fragment>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>

      {/* The second step, and only when the chosen model has one: which point of
          its training to talk to. Sibling rather than nested, so the list above
          stays a list of models — see `CheckpointSelector`. */}
      <CheckpointSelector
        modelId={parentId}
        value={value}
        onValueChange={(id) => {
          // The pairing is known for certain here, which spares the request that
          // would otherwise re-learn the parent of what was just clicked.
          record(id, parentId)
          onValueChange(id)
        }}
      />
    </>
  )
}
