/**
 * A message sent while a turn is already streaming, before it has been picked
 * up — see `hooks/use-chat.ts`'s `queued` and the `interject` / `interjected`
 * frames in `lib/chat-socket.ts`.
 *
 * Deliberately not a bubble. A real user turn fills `bg-secondary` precisely so
 * it reads as *said*; this one hasn't been, so it gets the opposite treatment —
 * muted text and a hairline rule, the same quiet register `eyebrow` and a rail
 * already carry elsewhere (see `docs/frontend/README.md`'s Sizing section). The
 * rule sits on the right, mirroring the trace's left rail, because this is
 * still on the user's side of the thread.
 */
export interface QueuedMessageProps {
  text: string
}

export function QueuedMessage({ text }: QueuedMessageProps) {
  return (
    <div className="flex w-full justify-end">
      <div className="max-w-[min(52ch,85%)] min-w-0 border-r border-border pr-3 text-right">
        <span className="tag text-muted-foreground/60">queued</span>
        <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{text}</p>
      </div>
    </div>
  )
}
