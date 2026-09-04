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
import { useCheckpointParent } from "@/hooks/use-checkpoint-parent"
import { useCheckpoints } from "@/hooks/use-checkpoints"
import { useModels, type Model } from "@/hooks/use-models"
import { THINK_BUDGETS, type ThinkBudget } from "@/lib/think-budget"

export interface ComposerPickerProps {
  model: string
  onModelChange: (id: string) => void
  budget: ThinkBudget
  onBudgetChange: (budget: ThinkBudget) => void
}

/**
 * What the next turn sends with: the model, which of its checkpoints (if any),
 * and how hard it should think — one popover rather than the two-to-three
 * separate triggers this used to be.
 *
 * `ModelSelector`, `CheckpointSelector` and `BudgetSelector` used to be three
 * components (the checkpoint one nested inside the model one as a sibling
 * trigger) so that, per the earlier design, "that list stays a list of
 * models." This supersedes that rationale, at the user's request: a model, its
 * checkpoint and the thinking budget are all "what to send this turn with,"
 * and the footer reads better as one control that opens to three clearly
 * separated sections than as two-to-three buttons in a row. The checkpoint
 * section still only appears for a model whose run left them, nested under
 * that model's own row rather than blended into the model list.
 *
 * Owns exactly the state and requests the three predecessors owned between
 * them — `useModels`, `useCheckpointParent`, `useCheckpoints`, and the
 * stale-persisted-id correction — none of that logic changed, only which
 * component renders it.
 */
export function ComposerPicker({
  model,
  onModelChange,
  budget,
  onBudgetChange,
}: ComposerPickerProps) {
  const { models, isLoading } = useModels()
  // The choice may name a checkpoint, which is deliberately in no model list —
  // so `model` alone cannot say which model is selected, and an id missing
  // from the list is not yet evidence of anything.
  const { parentId, isResolving, record } = useCheckpointParent(model, models)
  const { checkpoints } = useCheckpoints(parentId || null)

  // Default to the first model once the list has loaded — and correct a
  // persisted choice that no longer exists (deleted or unpublished since a
  // past session), which otherwise shows a blank trigger forever.
  //
  // Waits on `isResolving`: a checkpoint is absent from `models` exactly as a
  // deleted model is, and correcting before the two are told apart would drop
  // every checkpoint choice the moment it was made.
  React.useEffect(() => {
    if (models.length > 0 && !isResolving && !parentId) {
      onModelChange(models[0].id)
    }
  }, [models, isResolving, parentId, onModelChange])

  const selectedModel = models.find((m) => m.id === parentId)
  const selectedCheckpoint = checkpoints.find((c) => c.id === model)
  const onFinalWeights = model === parentId

  function chooseCheckpoint(id: string) {
    // The pairing is known for certain here, which spares the request that
    // would otherwise re-learn the parent of what was just clicked.
    record(id, parentId)
    onModelChange(id)
  }

  // Group models under their family, families sorted alphabetically — the
  // same grouping `ModelSelector` used.
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

  const modelLabel = selectedModel?.name ?? (isLoading ? "Loading…" : "Select model")
  const checkpointLabel = checkpoints.length === 0
    ? null
    : onFinalWeights
      ? "Final"
      : selectedCheckpoint
        ? `#${selectedCheckpoint.index}`
        : null
  const budgetOption = THINK_BUDGETS.find((b) => b.value === budget)
  // "Auto" is the absence of a choice, so the trigger names it only when it
  // isn't — see `lib/think-budget.ts`.
  const budgetLabel = budget === "auto" ? null : (budgetOption?.label ?? null)

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={
          <InputGroupButton
            aria-label="Model, checkpoint and thinking budget"
            disabled={isLoading || models.length === 0}
            className="gap-1.5"
          >
            <span>{modelLabel}</span>
            {checkpointLabel && (
              <span className="text-muted-foreground">· {checkpointLabel}</span>
            )}
            {budgetLabel && (
              <span className="text-muted-foreground">· {budgetLabel}</span>
            )}
            <ChevronDown className="text-muted-foreground" />
          </InputGroupButton>
        }
      />
      <DropdownMenuContent
        side="top"
        align="start"
        className="max-h-96 min-w-64 overflow-y-auto p-1"
      >
        {groups.map(([family, familyModels], i) => (
          <React.Fragment key={family}>
            {i > 0 && <DropdownMenuSeparator />}
            <DropdownMenuGroup>
              <DropdownMenuLabel className="eyebrow text-muted-foreground/80">
                {family}
              </DropdownMenuLabel>
              {familyModels.map((familyModel) => {
                // Marked against the *model* the choice belongs to, so a
                // checkpoint choice still shows which model it's under.
                const isActive = familyModel.id === parentId
                return (
                  <React.Fragment key={familyModel.id}>
                    <DropdownMenuItem
                      onClick={() => onModelChange(familyModel.id)}
                      className="gap-3 py-1.5"
                    >
                      <Check
                        className={cn(
                          "size-3.5 text-primary transition-opacity",
                          isActive ? "opacity-100" : "opacity-0"
                        )}
                      />
                      <span className={cn(isActive && "font-medium")}>
                        {familyModel.name}
                      </span>
                      <span className="ml-auto eyebrow text-muted-foreground/70">
                        {familyModel.tier}
                      </span>
                    </DropdownMenuItem>
                    {/* The second step, nested under the model it belongs to
                        rather than a flyout of its own, and only rendered for
                        the currently selected model whose run left
                        intermediates — see `useCheckpoints`. */}
                    {isActive && checkpoints.length > 0 && (
                      <DropdownMenuGroup>
                        <DropdownMenuItem
                          inset
                          onClick={() => chooseCheckpoint(familyModel.id)}
                          className="gap-3 py-1"
                        >
                          <Check
                            className={cn(
                              "size-3.5 text-primary transition-opacity",
                              onFinalWeights ? "opacity-100" : "opacity-0"
                            )}
                          />
                          <span
                            className={cn(
                              "text-sm",
                              onFinalWeights && "font-medium"
                            )}
                          >
                            Finished weights
                          </span>
                        </DropdownMenuItem>
                        {checkpoints.map((checkpoint) => {
                          const checkpointActive = checkpoint.id === model
                          return (
                            <DropdownMenuItem
                              key={checkpoint.id}
                              inset
                              onClick={() => chooseCheckpoint(checkpoint.id)}
                              className="gap-3 py-1"
                              // A checkpoint with no weights left on disk
                              // cannot answer. Its row outliving its directory
                              // is what `has_weights` reports.
                              disabled={!checkpoint.hasWeights}
                            >
                              <Check
                                className={cn(
                                  "size-3.5 text-primary transition-opacity",
                                  checkpointActive
                                    ? "opacity-100"
                                    : "opacity-0"
                                )}
                              />
                              <span
                                className={cn(
                                  "text-sm",
                                  checkpointActive && "font-medium"
                                )}
                              >
                                #{checkpoint.index}
                              </span>
                              <span className="ml-auto eyebrow text-muted-foreground/70 tabular-nums">
                                step {checkpoint.step.toLocaleString()}
                              </span>
                            </DropdownMenuItem>
                          )
                        })}
                      </DropdownMenuGroup>
                    )}
                  </React.Fragment>
                )
              })}
            </DropdownMenuGroup>
          </React.Fragment>
        ))}

        <DropdownMenuSeparator />
        {/* A different kind of choice from the two above it — how hard the
            model thinks rather than which model — so it reads as its own
            section rather than blending into the model list. */}
        <DropdownMenuGroup>
          <DropdownMenuLabel className="eyebrow text-muted-foreground/80">
            Thinking
          </DropdownMenuLabel>
          {THINK_BUDGETS.map((option) => {
            const isActive = option.value === budget
            return (
              <DropdownMenuItem
                key={option.value}
                onClick={() => onBudgetChange(option.value)}
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
                    {option.label}
                  </span>
                </span>
                <span className="ml-auto text-xs text-muted-foreground">
                  {option.hint}
                </span>
              </DropdownMenuItem>
            )
          })}
        </DropdownMenuGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
