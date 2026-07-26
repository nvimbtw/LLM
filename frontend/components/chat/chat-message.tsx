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
        <div className="min-w-0 max-w-[85%] rounded-2xl bg-card px-4 py-3 text-card-foreground ring-1 ring-border/50">
          <RichContent root={tree} />
        </div>
      </div>
    )
  }

  return (
    <div className="w-full min-w-0">
      <RichMessage content={message.content} streaming={streaming} />
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
        "text-muted-foreground/50 hover:text-foreground",
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
    figures.push(`${message.thinkingTokenCount.toLocaleString("en-US")} thinking`)
  }
  if (!message.modelName && figures.length === 0) return null

  return (
    <p className="flex flex-wrap items-baseline gap-x-2 text-xs text-muted-foreground/70">
      {message.modelName && (
        // The one part that's a name rather than a number, so it carries the weight.
        <span className="font-medium text-muted-foreground">{message.modelName}</span>
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
