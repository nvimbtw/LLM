"use client"

import * as React from "react"
import { ChevronDown, Sparkles } from "lucide-react"

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { InputGroupButton } from "@/components/ui/input-group"
import { useModels } from "@/hooks/use-models"

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
          {models.map((model) => (
            <DropdownMenuRadioItem key={model.id} value={model.id}>
              {model.name}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
