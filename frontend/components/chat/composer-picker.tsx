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
import { useComposerChoice } from "./composer-choice-provider"
import { useCheckpointParent } from "@/hooks/use-checkpoint-parent"
import { useModels } from "@/hooks/use-models"
import { useAuth } from "@/hooks/use-auth"
import { THINK_BUDGETS, type ThinkBudget } from "@/lib/think-budget"

export interface ComposerPickerProps {
  model: string
  onModelChange: (id: string) => void
  budget: ThinkBudget
  onBudgetChange: (budget: ThinkBudget) => void
}

/** Offers searchable published models and a separate thinking choice; resolves existing checkpoint identities before correcting stale persisted choices. */
export function ComposerPicker({
  model,
  onModelChange,
  budget,
  onBudgetChange,
}: ComposerPickerProps) {
  const { models, isLoading, error } = useModels()
  const { isLoading: choiceLoading } = useComposerChoice()
  const { isAdmin } = useAuth()
  const { parentId, isResolving } = useCheckpointParent(model, models)
  const [query, setQuery] = React.useState("")
  const [all, setAll] = React.useState(false)
  const published = models.filter((item) => item.published)
  const choices = (all && isAdmin ? models : published).filter((item) =>
    `${item.name} ${item.family}`
      .toLocaleLowerCase()
      .includes(query.toLocaleLowerCase())
  )
  React.useEffect(() => {
    const first = published[0] ?? (isAdmin ? models[0] : undefined)
    if (
      !error &&
      !choiceLoading &&
      !isLoading &&
      !isResolving &&
      !parentId &&
      first
    )
      onModelChange(first.id)
  }, [
    error,
    choiceLoading,
    isLoading,
    isResolving,
    parentId,
    published,
    models,
    isAdmin,
    onModelChange,
  ])
  const loading = choiceLoading || isLoading || isResolving
  const selected = models.find((item) => item.id === parentId)
  return (
    <div className="flex min-w-0 items-center gap-1">
      <DropdownMenu
        onOpenChange={(open) => {
          if (open) {
            setAll(false)
            setQuery("")
          }
        }}
      >
        <DropdownMenuTrigger
          render={
            <InputGroupButton
              aria-label="Choose model"
              disabled={loading || !models.length}
              aria-busy={loading}
              className="min-w-0 gap-1 text-xs"
            >
              <span className="max-w-[min(11rem,25vw)] truncate">
                {loading
                  ? "Loading models…"
                  : (selected?.name ?? "Select model")}
              </span>
              <ChevronDown className="size-3" />
            </InputGroupButton>
          }
        />
        <DropdownMenuContent
          side="top"
          align="start"
          className="w-72 max-w-[calc(100vw-2rem)] rounded-(--radius-composer) p-3 shadow-overlay"
        >
          <input
            aria-label="Search models"
            placeholder="Search models…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => event.stopPropagation()}
            className="mb-3 w-full border-0 border-b border-input bg-transparent py-2 text-xs outline-none focus:border-primary"
          />
          {isAdmin && (
            <div className="mb-2 flex gap-4 text-xs">
              <button
                type="button"
                aria-pressed={!all}
                className={!all ? "text-primary" : "text-muted-foreground"}
                onClick={() => setAll(false)}
              >
                Published
              </button>
              <button
                type="button"
                aria-pressed={all}
                className={all ? "text-primary" : "text-muted-foreground"}
                onClick={() => setAll(true)}
              >
                All models
              </button>
            </div>
          )}
          <div className="max-h-64 overflow-y-auto">
            {choices.map((item) => (
              <DropdownMenuItem
                key={item.id}
                onClick={() => onModelChange(item.id)}
                className="gap-2 py-2 text-xs"
              >
                <Check
                  className={
                    item.id === parentId
                      ? "size-3 text-primary"
                      : "size-3 opacity-0"
                  }
                />
                <span className="min-w-0">
                  <span className="block truncate">{item.name}</span>
                  <span className="text-2xs text-muted-foreground">
                    {item.family}
                    {item.tier ? ` · ${item.tier}` : ""}
                  </span>
                </span>
              </DropdownMenuItem>
            ))}
            {!choices.length && (
              <p className="py-3 text-xs text-muted-foreground">
                No matching models
              </p>
            )}
          </div>
        </DropdownMenuContent>
      </DropdownMenu>
      <DropdownMenu>
        <DropdownMenuTrigger
          render={
            <InputGroupButton
              aria-label="Thinking budget"
              disabled={choiceLoading}
              className="gap-1 text-xs text-muted-foreground"
            >
              {THINK_BUDGETS.find((item) => item.value === budget)?.label ??
                "Auto"}
              <ChevronDown className="size-3" />
            </InputGroupButton>
          }
        />
        <DropdownMenuContent
          side="top"
          align="start"
          className="min-w-48 rounded-(--radius-composer) p-1.5 shadow-overlay"
        >
          {THINK_BUDGETS.map((item) => (
            <DropdownMenuItem
              key={item.value}
              onClick={() => onBudgetChange(item.value)}
              className="gap-2 text-xs"
            >
              <Check
                className={
                  item.value === budget
                    ? "size-3 text-primary"
                    : "size-3 opacity-0"
                }
              />
              {item.label}
              <span className="ml-auto text-2xs text-muted-foreground">
                {item.hint}
              </span>
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}
