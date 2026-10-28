import { Input, type InputProps } from "./input"
import { cn } from "@/lib/utils"

export interface EditableTitleProps extends InputProps {
  "aria-label": string
}

/** Presents an unboxed page title and reveals a small Edit cue on field hover or keyboard focus, preserving its input/ref contract. */
export function EditableTitle({ className, ...props }: EditableTitleProps) {
  return (
    <span className="group/editable relative flex min-w-0 flex-1">
      <Input
        {...props}
        variant="inline"
        className={cn(
          "h-auto min-w-0 flex-1 py-0 pr-9 font-display text-3xl font-semibold tracking-tight md:text-3xl",
          className
        )}
      />
      {!props.readOnly && !props.disabled && (
        <span
          aria-hidden="true"
          className="pointer-events-none absolute top-1/2 right-0 -translate-y-1/2 font-sans text-2xs font-normal tracking-normal text-muted-foreground opacity-0 transition-opacity group-focus-within/editable:text-primary group-focus-within/editable:opacity-100 group-hover/editable:opacity-100 motion-reduce:transition-none"
        >
          Edit
        </span>
      )}
    </span>
  )
}
