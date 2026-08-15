"use client"

import * as React from "react"
import { ThumbsDown, ThumbsUp } from "lucide-react"

import type { ChatMessage } from "@/hooks/use-chat"
import type { Feedback } from "@/lib/feedback"
import { parseUserMarkdown } from "@/lib/user-parser"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { RichContent, RichMessage } from "./rich-message"

export function ChatMessageBubble({
  message,
  streaming = false,
  onRate,
}: {
  message: ChatMessage
  streaming?: boolean
  // Absent on a reply that can't be rated (no server id yet).
  onRate?: (feedback: Feedback) => void
}) {
  // User messages use a markdown subset; assistant messages use the model's markup.
  if (message.role === "user") {
    const tree = parseUserMarkdown(message.content)
    return (
      <div className="flex w-full justify-end">
        {/* Capped in `ch` as well as `%`: a percentage alone means a two-word
            question and a five-line one get wildly different shapes, and the long
            one runs to a measure nobody wants to read right-aligned.

            A fill and no outline. It used to carry both — a 1px border *and* a
            card ground — which is how the app draws an input or a panel, so a
            question the reader typed came back to them looking like a control
            they could edit. One quiet step off the paper is enough to say
            "someone said this", and it is the only device the assistant side
            doesn't also have.

            The bottom-right corner is tightened to a fraction of the others. That
            asymmetry is the whole reason it reads as *said by someone* rather than
            as a rounded rectangle — a tail, without drawing a tail. */}
        <div className="max-w-[min(52ch,85%)] min-w-0 rounded-2xl rounded-br-sm bg-secondary px-[1.05em] py-[0.7em] text-secondary-foreground">
          <RichContent root={tree} className="text-[0.95rem] leading-[1.6]" />
        </div>
      </div>
    )
  }

  return (
    <div className="w-full min-w-0">
      {/* No bubble on the assistant side: the reply is the page, not a quote on
          it. It gets the larger size of the two — this is the text that actually
          gets read.

          It does get a measure. The thread column is sized for the composer and
          the user bubbles, and at 16px that left the reply running to roughly 75
          characters a line — the outer edge of what stays readable, and past the
          point where the eye reliably finds the start of the next line. 68ch is
          the one declaration in this file that does the most for legibility.

          Left-aligned rather than centred in the column: the replies then share
          a left edge with the byline under them and the status row below the
          thread, so the short ones don't wander. */}
      <RichMessage
        content={message.content}
        streaming={streaming}
        className="max-w-[68ch] text-base leading-relaxed"
      />
      {/* Nothing under a reply that is still being written: the byline's figures
          aren't known yet, and a thumbs on half a reply would rate text the user
          hasn't finished reading. */}
      {!streaming && (
        <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 px-1">
          <Byline message={message} />
          {message.id && onRate && (
            <FeedbackButtons feedback={message.feedback} onRate={onRate} />
          )}
        </div>
      )}
    </div>
  )
}

// Thumbs up/down on a reply. Deliberately quiet — muted and borderless until
// hovered or chosen — because it sits under every single reply and a pair of
// prominent buttons there would compete with the text for attention.
//
// The chosen side stays filled after clicking, so the rating reads as a state the
// reply now carries rather than as an action that has been fired and forgotten.
function FeedbackButtons({
  feedback,
  onRate,
}: {
  feedback?: Feedback
  onRate: (feedback: Feedback) => void
}) {
  return (
    <div className="flex items-center gap-0.5">
      <FeedbackButton
        icon={ThumbsUp}
        // "Good reply"/"Bad reply" rather than "Like"/"Dislike": the question is
        // whether the model answered well, not whether the user enjoyed it.
        label="Good reply"
        active={feedback === "up"}
        onClick={() => onRate("up")}
      />
      <FeedbackButton
        icon={ThumbsDown}
        label="Bad reply"
        active={feedback === "down"}
        onClick={() => onRate("down")}
      />
    </div>
  )
}

function FeedbackButton({
  icon: Icon,
  label,
  active,
  onClick,
}: {
  icon: typeof ThumbsUp
  label: string
  active: boolean
  onClick: () => void
}) {
  return (
    <Button
      variant="ghost"
      size="icon-xs"
      // Announces the rating as a toggle, so a screen reader says whether this
      // reply is currently rated rather than just naming the button.
      aria-pressed={active}
      aria-label={label}
      title={active ? `${label} — click to undo` : label}
      onClick={onClick}
      className={cn(
        "text-muted-foreground hover:text-foreground",
        active && "text-foreground"
      )}
    >
      <Icon
        // Filling the chosen thumb is what makes the state readable at a glance;
        // colour alone would be too subtle at this size and would carry the whole
        // meaning on hue.
        className={active ? "fill-current" : undefined}
      />
    </Button>
  )
}

// Attribution under a reply: which model wrote it, then what it cost.
//
// Either half can be missing, which is why this isn't one string. A reply loaded from
// history has a model name but no token counts (the counts are tallied client-side as
// tokens arrive and aren't persisted), while a reply that just streamed has counts and
// only gains its name when `onDone` refetches it. Rendering the two independently means
// each shows whatever is actually known.
//
// The mid-stream guard lives in the caller now, since the feedback buttons share it.
function Byline({ message }: { message: ChatMessage }) {
  const figures: string[] = []
  if (message.tokenCount !== undefined) {
    figures.push(`${message.tokenCount.toLocaleString("en-US")} tokens`)
  }
  if (message.thinkingTokenCount) {
    figures.push(
      `${message.thinkingTokenCount.toLocaleString("en-US")} thinking`
    )
  }
  if (!message.modelName && figures.length === 0) return null

  return (
    <p className="flex flex-wrap items-baseline gap-x-2 text-xs text-muted-foreground">
      {message.modelName && (
        // The one part that's a name rather than a number, so it carries the weight.
        <span className="font-medium text-muted-foreground">
          {message.modelName}
        </span>
      )}
      {figures.map((figure, i) => (
        <React.Fragment key={figure}>
          {(i > 0 || message.modelName) && (
            <span aria-hidden className="text-muted-foreground/35">
              ·
            </span>
          )}
          <span className="tabular-nums">{figure}</span>
        </React.Fragment>
      ))}
    </p>
  )
}
