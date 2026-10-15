"use client"

import * as React from "react"
import { Checkbox as CheckboxPrimitive } from "@base-ui/react/checkbox"
import { CheckIcon } from "lucide-react"

import { cn } from "@/lib/utils"

export interface CheckboxProps extends CheckboxPrimitive.Root.Props {
  variant?: "default" | "block"
}

/** Edits a boolean using a compact mark or a fully clickable labeled block. Selected blocks use the theme accent without changing their text contrast. */
function Checkbox({
  className,
  variant = "default",
  children,
  ...props
}: CheckboxProps) {
  return (
    <CheckboxPrimitive.Root
      data-slot="checkbox"
      className={cn(
        "peer shrink-0 border transition-colors outline-none focus-visible:ring-3 focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50 data-disabled:cursor-not-allowed data-disabled:opacity-50",
        variant === "block"
          ? "flex w-full cursor-pointer items-start justify-between gap-4 rounded-(--radius-control) border-border/60 bg-transparent p-4 text-left text-foreground data-checked:border-primary/40 data-checked:bg-primary/10"
          : "size-4 rounded-(--radius-marker) border-input bg-transparent shadow-raised data-checked:border-primary data-checked:bg-primary data-checked:text-primary-foreground",
        className
      )}
      {...props}
    >
      {variant === "block" ? (
        <>
          {children}
          <span className="mt-0.5 flex size-4 shrink-0 items-center justify-center text-primary">
            <CheckboxPrimitive.Indicator data-slot="checkbox-indicator">
              <CheckIcon className="size-3.5" strokeWidth={3} />
            </CheckboxPrimitive.Indicator>
          </span>
        </>
      ) : (
        <CheckboxPrimitive.Indicator
          data-slot="checkbox-indicator"
          className="flex items-center justify-center text-current"
        >
          <CheckIcon className="size-3.5" strokeWidth={3} />
        </CheckboxPrimitive.Indicator>
      )}
    </CheckboxPrimitive.Root>
  )
}

export { Checkbox }
