"use client"

import { CornerLeftUp } from "lucide-react"

import { cn } from "@/lib/utils"
import { revealMessage, useMessageRef } from "./message-refs"

/**
 * `<|reply|>message-id<|/reply|>` — this passage answers an earlier message.
 *
 * Rendered as the quoted header you get in a messaging app: who said it, one line
 * of what they said, tucked above the text that responds to it. Compact on
 * purpose — it is a pointer, and the reply itself is what should be read.
 *
 * Clicking it goes to the message. A reference you cannot follow is only half a
 * reference: the excerpt says roughly which message is meant, and the click is
 * what settles it when a conversation has three similar-looking questions in it.
 *
 * The arrow points **up**, toward the thing being referenced — not the outward
 * "reply" arrow, which reads as pointing away from the message you are being
 * sent to.
 */
export interface MessageReplyRefProps {
  id: string
}

export function MessageReplyRef({ id }: MessageReplyRefProps) {
  const ref = useMessageRef(id)

  const body = (
    <>
      <CornerLeftUp
        aria-hidden
        className="mt-0.5 size-3.5 shrink-0 text-muted-foreground"
      />
      <span className="min-w-0">
        <span className="block text-xs font-medium tracking-[0.06em] text-muted-foreground uppercase">
          {ref
            ? ref.role === "user"
              ? "Replying to you"
              : "Replying to itself"
            : "Replying to an earlier message"}
        </span>
        {ref ? (
          <span className="mt-0.5 line-clamp-2 block text-muted-foreground">
            {ref.excerpt}
          </span>
        ) : (
          // Not the raw id: it means nothing to a reader, and an opaque string
          // where a quote should be looks like a bug rather than a missing target. The
          // id stays in the tooltip, which is where it is useful — debugging a
          // model that emits references to messages that never existed.
          <span className="mt-0.5 block text-muted-foreground italic">
            No longer in this conversation
          </span>
        )}
      </span>
    </>
  )

  const className = cn(
    "my-3 flex w-full items-start gap-2 rounded-md border border-l-2 border-border border-l-primary bg-muted/50 py-2 pr-3 pl-2.5 text-left text-sm"
  )

  // Unresolvable references are not buttons. Making one look clickable and then
  // do nothing is worse than it plainly not being clickable.
  if (!ref) {
    return (
      <div className={className} title={`Message ${id}`}>
        {body}
      </div>
    )
  }

  return (
    <button
      type="button"
      onClick={() => revealMessage(id)}
      className={cn(
        className,
        "cursor-pointer transition-colors hover:border-l-primary hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
      )}
    >
      {body}
    </button>
  )
}
