import { cn } from "@/lib/utils"

/**
 * The model's raw reasoning (the `<|thinking|>` block), shown above the reply
 * when "Show reasoning" is on. Deliberately quiet — a hairline left rail and
 * muted text — so it reads as a margin note next to the answer, not a second
 * answer. Streams in place, so it also renders partial reasoning mid-thought.
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
        "mb-2 border-l-2 border-border/70 pl-3 text-muted-foreground",
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
