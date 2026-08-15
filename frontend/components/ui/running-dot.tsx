import { cn } from "@/lib/utils"

/**
 * A small pulsing dot: "this is running", for the places the throbber won't fit.
 *
 * The throbber is an 84px figure. That is the right mark for a panel whose subject
 * *is* the waiting — a chat thread with nothing in it yet, a benchmark's empty
 * state, the training hero — and the wrong one beside a line of text, inside a
 * button, or in a sidebar row. Those places need the fact of activity, not
 * something to look at, and a dot is the whole of that fact.
 *
 * Colour comes from `className` (tailwind-merge lets a caller's `bg-*` win), which
 * is how the sidebar tints it to its own foreground.
 */
export interface RunningDotProps {
  /** Announced in place of the dot, which has no text of its own. */
  label?: string
  className?: string
}

export function RunningDot({ label = "Running", className }: RunningDotProps) {
  return (
    <span
      role="status"
      aria-label={label}
      className={cn(
        "inline-block size-2 shrink-0 animate-pulse rounded-full bg-primary",
        className
      )}
    />
  )
}
