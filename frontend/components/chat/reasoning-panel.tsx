import { cn } from "@/lib/utils"

/**
 * A single `<|thinking|>` block, rendered inline at the point in the reply
 * where the model reasoned (there may be several per message). Deliberately
 * quiet — a hairline left rail and muted text — so it reads as a margin note
 * next to the answer, not a second answer. Streams in place, so it also
 * renders partial reasoning mid-thought.
 */
export function ReasoningPanel({
  text,
  className,
}: {
  text: string
  className?: string
}) {
  if (!text) return null
  return (
    <div
      className={cn(
        // Symmetric vertical margin for calm breathing room above and below the
        // reply text. Tailwind v4's `space-y` uses a zero-specificity `:where()`
        // selector, so this `my` reliably overrides the parent's flow gap.
        "my-4 border-l-2 border-border/70 pl-3 text-muted-foreground",
        className
      )}
    >
      <p className="mb-1 font-mono text-[10px] tracking-wide uppercase text-muted-foreground/70">
        Reasoning
      </p>
      <p className="text-sm leading-relaxed whitespace-pre-wrap">{text}</p>
    </div>
  )
}
