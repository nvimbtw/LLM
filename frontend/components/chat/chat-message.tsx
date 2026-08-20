"use client"

import * as React from "react"
import {
  Check,
  Copy,
  Pencil,
  RotateCcw,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react"

import type { ChatMessage } from "@/hooks/use-chat"
import type { Feedback } from "@/lib/feedback"
import { parseUserMarkdown } from "@/lib/user-parser"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { RichContent, RichMessage } from "./rich-message"
import { Trace } from "./trace"

export function ChatMessageBubble({
  message,
  streaming = false,
  onRate,
  onEdit,
  onResend,
  busy = false,
}: {
  message: ChatMessage
  streaming?: boolean
  // Absent on a reply that can't be rated (no server id yet).
  onRate?: (feedback: Feedback) => void
  // User messages only: put this question back in the composer to change it, and
  // ask it again as it stands. Both append a new turn — see `UserActions`.
  onEdit?: (content: string) => void
  onResend?: (content: string) => void
  /** A turn is already in flight, so nothing may start another one. */
  busy?: boolean
}) {
  // User messages use a markdown subset; assistant messages use the model's markup.
  if (message.role === "user") {
    const tree = parseUserMarkdown(message.content)
    return (
      // `group` so the actions below can dim until the turn is hovered, and
      // `flex-col items-end` so they hang under the bubble's right edge rather
      // than under the column's.
      <div className="group flex w-full flex-col items-end">
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
        <div className="max-w-[min(52ch,85%)] min-w-0 rounded-(--radius-bubble) rounded-br-(--radius-marker) bg-secondary px-[1.05em] py-[0.7em] text-secondary-foreground">
          {/* `text-base`, the scale's reading step. This was `text-[0.95rem]`
              with a hand-set 1.6 leading — 15.2px, which is 0.8px off the size
              the scale already defines *as* the size for reading, with a
              line-height and tracking chosen to match it. A chat answer is the
              longest prose in the app; if anything is set at the reading size,
              it is this. */}
          <RichContent root={tree} className="text-base" />
        </div>

        <UserActions
          content={message.content}
          onEdit={onEdit}
          onResend={onResend}
          busy={busy}
        />
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
      {/* A turn that reached for a tool ran as more than one generation, and
          the whole of the first plus what came back is its working. It sits
          above the answer at the same measure, so the two share a left edge and
          the trace reads as a note in the margin of the reply rather than as a
          second reply. A turn that called nothing has no `working` and renders
          exactly as it always did — its reasoning stays inline, in the place the
          model actually stopped to think. */}
      {message.working !== undefined && (
        <Trace
          working={message.working}
          answers={message.toolAnswers ?? []}
          elapsedSecs={message.elapsedSecs}
          streaming={streaming}
          className="max-w-[68ch] text-base leading-relaxed"
        />
      )}
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

/**
 * What you can do with a question you already asked: copy it, edit it, ask it
 * again.
 *
 * **Edit and resend both append a new turn** rather than rewriting the one that
 * is there. That is a deliberate limit and not an oversight: the thread is the
 * transcript the model is prompted with, and there is no route that removes a
 * stored message (`backend/crates/api/src/routes/chats.rs` has create, rename
 * and delete-the-whole-chat, and nothing finer). Editing in place would mean
 * either lying about what was sent or silently discarding everything after it.
 * Asking again, out loud, at the end, is the honest version of the same action —
 * and it is what "resend" says on the label.
 *
 * Dimmed rather than hidden, for the reason written on the code block's copy
 * button in `rich-message.tsx`: hover is the obvious way to reveal a control and
 * the wrong one, because a touch screen has no hover and the affordance would
 * simply not exist there.
 */
function UserActions({
  content,
  onEdit,
  onResend,
  busy,
}: {
  content: string
  onEdit?: (content: string) => void
  onResend?: (content: string) => void
  busy: boolean
}) {
  const [copied, setCopied] = React.useState(false)

  // The raw markdown, not the rendered text: what goes to the clipboard should
  // be what you typed, so pasting it back into the composer asks the same
  // question rather than a flattened version of it.
  async function copy() {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard unavailable — ignore.
    }
  }

  return (
    <div className="mt-1.5 flex items-center gap-0.5 pr-0.5 opacity-45 transition-opacity duration-150 group-hover:opacity-100 focus-within:opacity-100">
      <MessageAction
        icon={copied ? Check : Copy}
        label={copied ? "Copied" : "Copy message"}
        onClick={copy}
      />
      {onEdit && (
        <MessageAction
          icon={Pencil}
          // Named for what it does, not for what it looks like it does: this
          // loads the question into the composer, and sending it asks a new
          // turn. "Edit" alone would promise the message is about to change.
          label="Edit and ask again"
          disabled={busy}
          onClick={() => onEdit(content)}
        />
      )}
      {onResend && (
        <MessageAction
          icon={RotateCcw}
          label="Ask again"
          disabled={busy}
          onClick={() => onResend(content)}
        />
      )}
    </div>
  )
}

function MessageAction({
  icon: Icon,
  label,
  onClick,
  disabled,
}: {
  icon: typeof Copy
  label: string
  onClick: () => void
  disabled?: boolean
}) {
  return (
    <Button
      variant="ghost"
      size="icon-xs"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
      className="text-muted-foreground hover:text-foreground"
    >
      <Icon />
    </Button>
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
