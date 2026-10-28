import { Input, type InputProps } from "./input"
import { cn } from "@/lib/utils"

export interface EditableTitleProps extends InputProps {
  "aria-label": string
}

/** Presents a page title as an accessible unboxed field using shared typography and caret-only editing. */
export function EditableTitle({ className, ...props }: EditableTitleProps) {
  return (
    <Input
      {...props}
      variant="inline"
      className={cn(
        "h-auto min-w-0 flex-1 py-0 font-display text-3xl font-semibold tracking-tight md:text-3xl",
        className
      )}
    />
  )
}
