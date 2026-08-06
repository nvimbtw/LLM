"use client"

import * as React from "react"
import { Check, ChevronDown } from "lucide-react"

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { InputGroupButton } from "@/components/ui/input-group"
import { cn } from "@/lib/utils"
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
            {selected?.name ?? (isLoading ? "Loading…" : "Select model")}
            <ChevronDown className="text-muted-foreground" />
          </InputGroupButton>
        }
      />
      {/* Two levels: families up top, hover one to reveal its models. Kept spare
          — hairline dividers between families, a muted count on the right, the
          active family and model marked but never shouted. */}
      <DropdownMenuContent side="top" align="start" className="min-w-52 p-1">
        {groups.map(([family, familyModels]) => {
          const activeFamily = familyModels.some((m) => m.id === value)
          return (
            <DropdownMenuSub key={family}>
              <DropdownMenuSubTrigger
                className={cn(
                  "gap-6 py-1.5",
                  activeFamily ? "text-foreground" : "text-muted-foreground"
                )}
              >
                <span className="font-medium tracking-tight">{family}</span>
                <span className="ml-auto text-xs text-muted-foreground tabular-nums">
                  {familyModels.length}
                </span>
              </DropdownMenuSubTrigger>
              <DropdownMenuSubContent className="min-w-56">
                {familyModels.map((model) => {
                  const isActive = model.id === value
                  return (
                    <DropdownMenuItem
                      key={model.id}
                      onClick={() => onValueChange(model.id)}
                      className="gap-6 py-1.5"
                    >
                      <span className="flex items-center gap-2">
                        <Check
                          className={cn(
                            "size-3.5 text-primary transition-opacity",
                            isActive ? "opacity-100" : "opacity-0"
                          )}
                        />
                        <span className={cn(isActive && "font-medium")}>
                          {model.name}
                        </span>
                      </span>
                      <span className="ml-auto text-xs font-medium tracking-[0.12em] text-muted-foreground uppercase">
                        {model.tier}
                      </span>
                    </DropdownMenuItem>
                  )
                })}
              </DropdownMenuSubContent>
            </DropdownMenuSub>
          )
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
