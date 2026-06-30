"use client"

import * as React from "react"
import { ChevronDown, Sparkles } from "lucide-react"

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { InputGroupButton } from "@/components/ui/input-group"
import { useModels, type Model } from "@/hooks/use-models"

export function ModelSelector({
  value,
  onValueChange,
}: {
  value: string
  onValueChange: (value: string) => void
}) {
  const { models, isLoading } = useModels()

  // Default to the first model once the list has loaded.
  React.useEffect(() => {
    if (!value && models.length > 0) {
      onValueChange(models[0].id)
    }
  }, [value, models, onValueChange])

  const selected = models.find((m) => m.id === value)

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
    <DropdownMenu>
      <DropdownMenuTrigger
        render={
          <InputGroupButton
            aria-label="Select model"
            disabled={isLoading || models.length === 0}
          >
            <Sparkles />
            {selected?.name ?? (isLoading ? "Loading…" : "Select model")}
            <ChevronDown />
          </InputGroupButton>
        }
      />
      <DropdownMenuContent side="top" align="start" className="w-auto min-w-44">
        <DropdownMenuRadioGroup value={value} onValueChange={onValueChange}>
          {groups.map(([family, familyModels], i) => (
            <React.Fragment key={family}>
              {i > 0 && <DropdownMenuSeparator />}
              <DropdownMenuLabel className="text-xs text-muted-foreground">
                {family}
              </DropdownMenuLabel>
              {familyModels.map((model) => (
                <DropdownMenuRadioItem key={model.id} value={model.id}>
                  {model.name}
                </DropdownMenuRadioItem>
              ))}
            </React.Fragment>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
