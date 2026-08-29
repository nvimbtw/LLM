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
import { useCheckpoints } from "@/hooks/use-checkpoints"
import { cn } from "@/lib/utils"

/**
 * Which point of a model's training to talk to: its finished weights, or one of
 * the checkpoints the run left on the way there.
 *
 * A control of its own beside the model picker rather than a flyout hanging off
 * it, and **absent entirely** for a model whose run left none — which is most of
 * them. That is the whole design: a checkpoint is offered *under* the model it
 * belongs to and never as a peer of it, so the model list stays the list of
 * models however many intermediates the runs behind it left. It is the same
 * two-step the trainer's `ResumePicker` uses, for the same reason.
 */
export interface CheckpointSelectorProps {
  /** The whole model whose intermediates are on offer. Empty renders nothing. */
  modelId: string
  /** The current choice: the model's own id, or one of its checkpoints'. */
  value: string
  onValueChange: (id: string) => void
}

export function CheckpointSelector({
  modelId,
  value,
  onValueChange,
}: CheckpointSelectorProps) {
  const { checkpoints, isLoading } = useCheckpoints(modelId || null)

  // Nothing to choose between is not a choice. Rendering an empty or
  // single-entry menu would put a control on every conversation to say that the
  // one model involved has no other version of itself.
  if (!modelId || isLoading || checkpoints.length === 0) return null

  const selected = checkpoints.find((c) => c.id === value)
  const onFinal = value === modelId

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={
          <InputGroupButton aria-label="Select checkpoint">
            {selected ? `#${selected.index}` : "Final"}
            <ChevronDown className="text-muted-foreground" />
          </InputGroupButton>
        }
      />
      <DropdownMenuContent
        side="top"
        align="start"
        className="max-h-72 min-w-56 overflow-y-auto p-1"
      >
        {/* Two groups with a rule between them, the arrangement `ModelSelector`
            uses: a label is a *group's* label — Base UI reads it through
            `MenuGroupContext` and throws without one — and a separator belongs
            between groups rather than inside one.

            The finished weights are their own group because that is what the
            model ordinarily means, not the last entry in a list of saves. */}
        <DropdownMenuGroup>
          <DropdownMenuItem
            onClick={() => onValueChange(modelId)}
            className="gap-3 py-1.5"
          >
            <Check
              className={cn(
                "size-3.5 text-primary transition-opacity",
                onFinal ? "opacity-100" : "opacity-0"
              )}
            />
            <span className={cn(onFinal && "font-medium")}>
              Finished weights
            </span>
          </DropdownMenuItem>
        </DropdownMenuGroup>
        <DropdownMenuSeparator />
        <DropdownMenuGroup>
          <DropdownMenuLabel className="eyebrow text-muted-foreground/80">
            Checkpoints
          </DropdownMenuLabel>
          {checkpoints.map((checkpoint) => {
            const isActive = checkpoint.id === value
            return (
              <DropdownMenuItem
                key={checkpoint.id}
                onClick={() => onValueChange(checkpoint.id)}
                className="gap-3 py-1.5"
                // A checkpoint with no weights left on disk cannot answer. Its
                // row outliving its directory is what `has_weights` reports.
                disabled={!checkpoint.hasWeights}
              >
                <Check
                  className={cn(
                    "size-3.5 text-primary transition-opacity",
                    isActive ? "opacity-100" : "opacity-0"
                  )}
                />
                <span className={cn(isActive && "font-medium")}>
                  #{checkpoint.index}
                </span>
                <span className="ml-auto eyebrow text-muted-foreground/70 tabular-nums">
                  step {checkpoint.step.toLocaleString()}
                </span>
              </DropdownMenuItem>
            )
          })}
        </DropdownMenuGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
