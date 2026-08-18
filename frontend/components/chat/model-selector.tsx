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
              <DropdownMenuLabel className="text-[11px] font-medium tracking-[0.1em] text-muted-foreground/80 uppercase">
                {family}
              </DropdownMenuLabel>
              {familyModels.map((model) => {
                const isActive = model.id === value
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
                    <span className="ml-auto text-[11px] font-medium tracking-[0.1em] text-muted-foreground/70 uppercase">
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
  )
}
