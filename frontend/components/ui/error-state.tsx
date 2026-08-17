import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

/**
 * The two ways this app reports a failure, and the only two.
 *
 * They were spelled by hand at fourteen call sites and had drifted into three
 * different objects: a bordered tint box (four copies, two radii, two
 * paddings), a bare red paragraph that left a section collapsed to one line of
 * text where a column of panels should be, and one panel written for the
 * tokenizer list. A reader met all three within a few clicks and had to work out
 * each time whether something had broken or was merely empty.
 *
 * The split is by *who can act on it*:
 *
 *  - `FormError` — a submission was rejected. It belongs against the control
 *    that produced it, it is read immediately, and it is the one place red
 *    earns its keep.
 *  - `LoadError` — a section could not fetch what it needed. Nothing the reader
 *    typed caused it, so it is not set in red; it takes the footprint of the
 *    content it replaces and offers the only useful action, which is to retry.
 */

export interface FormErrorProps {
  children: React.ReactNode
  className?: string
}

/**
 * Inline, against the form that produced it. `role="alert"` because it appears
 * in response to something the reader just did and has to be announced.
 */
export function FormError({ children, className }: FormErrorProps) {
  return (
    <p
      role="alert"
      className={cn(
        "rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive",
        className
      )}
    >
      {children}
    </p>
  )
}

export interface LoadErrorProps {
  /** One line, in plain words: what did not load. */
  title?: React.ReactNode
  /** Optional second line — what it means, or what is safe to assume. */
  hint?: React.ReactNode
  /** Rendered as "Try again" when the caller can refetch. */
  onRetry?: () => void
  className?: string
}

export function LoadError({
  title = "That didn't load.",
  hint,
  onRetry,
  className,
}: LoadErrorProps) {
  return (
    <div
      role="alert"
      className={cn(
        // The footprint of the content it stands in for, so a failed section
        // still occupies its column instead of collapsing the page around it.
        "rounded-md border border-border px-5 py-10 text-center",
        className
      )}
    >
      <p className="text-sm">{title}</p>
      {hint && (
        <p className="mx-auto mt-1 max-w-sm text-xs text-muted-foreground">
          {hint}
        </p>
      )}
      {onRetry && (
        <Button variant="outline" size="sm" className="mt-5" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  )
}
