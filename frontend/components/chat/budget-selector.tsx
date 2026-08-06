"use client"

import * as React from "react"
import { Check, ChevronDown } from "lucide-react"

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { InputGroupButton } from "@/components/ui/input-group"
import { cn } from "@/lib/utils"

/**
 * How long the model should reason before answering.
 *
 * `auto` is the absence of a choice, not a fourth level: it sends no budget at
 * all, so the model answers the way it did before budgets existed. The three
 * named levels are the ones the SFT corpus is labelled with — the wire values
 * match the labels in `raw/sft/think-*.jsonl` exactly, because the string the
 * composer sends becomes the marker the model was trained under.
 */
export type ThinkBudget = "auto" | "low" | "med" | "high"

const BUDGETS: { value: ThinkBudget; label: string; hint: string }[] = [
  { value: "auto", label: "Auto", hint: "Let the model decide" },
  { value: "low", label: "Low", hint: "Answer straight away" },
  { value: "med", label: "Medium", hint: "Weigh one alternative" },
  { value: "high", label: "High", hint: "Work it through properly" },
]

export function BudgetSelector({
  value,
  onValueChange,
}: {
  value: ThinkBudget
  onValueChange: (value: ThinkBudget) => void
}) {
  const selected = BUDGETS.find((b) => b.value === value) ?? BUDGETS[0]

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={
          <InputGroupButton aria-label="Select thinking budget">
            {/* "Thinking" is stated only when a level is chosen. On Auto the
                control names itself, so the row doesn't carry a word that
                describes the default state. */}
            {value === "auto" ? "Thinking" : `Thinking: ${selected.label}`}
            <ChevronDown className="text-muted-foreground" />
          </InputGroupButton>
        }
      />
      <DropdownMenuContent side="top" align="start" className="min-w-52 p-1">
        {BUDGETS.map((budget) => {
          const isActive = budget.value === value
          return (
            <DropdownMenuItem
              key={budget.value}
              onClick={() => onValueChange(budget.value)}
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
                  {budget.label}
                </span>
              </span>
              <span className="ml-auto text-xs text-muted-foreground">
                {budget.hint}
              </span>
            </DropdownMenuItem>
          )
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
