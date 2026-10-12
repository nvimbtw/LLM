"use client"

import * as React from "react"

import { cn } from "@/lib/utils"
import { hasVisibleContent, isInThinkingBlock } from "@/lib/message-parser"
import type { Attachment } from "@/lib/attachments"
import { useChat } from "@/hooks/use-chat"
import { useLocalPlan } from "@/hooks/use-local-plan"
import { ChatComposer } from "./chat-composer"
import { useChats } from "./chats-provider"
import { useComposerChoice } from "./composer-choice-provider"

/** How close (px) to the bottom the user must be for auto-scroll to engage. */
const STICK_TO_BOTTOM_THRESHOLD_PX = 120
import { AttachmentViewer } from "./attachment-viewer"
import { ChatMessageBubble } from "./chat-message"
import { Throbber } from "@/components/ui/throbber"
import { LoadError } from "@/components/ui/error-state"
import { ThinkingLabel } from "./thinking-label"
import { callInFlight } from "./trace"
import { ToolLabel } from "./tool-label"
import { QueuedMessage } from "./queued-message"
import { TokenFlow, type TokenFlowProps } from "./token-flow"
import { TurnClock, workedFor } from "./turn-clock"

/** The join between two things in the status row. One element, reused: it is
 * punctuation, and punctuation that differs between two commas in a sentence is
 * a mistake somewhere. */
const RULE = (
  <span aria-hidden className="-mx-1 shrink-0 text-border">
    |
  </span>
)

export function ChatThread({ chatId }: { chatId: string }) {
  const localPlan = useLocalPlan(chatId)
  const {
    messages,
    isLoading,
    isSending,
    queuedAhead,
    queued,
    error,
    sendMessage,
    cancel,
    rateMessage,
    prunedTokens,
    pruneContext,
    tokenArt,
  } = useChat(chatId)
  const { conversations, renameChat } = useChats()
  // Resending asks the same question with whatever the composer is currently set
  // to — the model and budget are a property of the ask, not of the message
  // being repeated, so a re-ask picks up a model you switched to since.
  const { model, budget } = useComposerChoice()
  const chatSummary = conversations.find((c) => c.id === chatId)
  // What "edit and ask again" has loaded into the composer. A fresh object each
  // time, so editing the same message twice loads it twice — see ChatComposer.
  const [draft, setDraft] = React.useState<{ text: string } | undefined>()
  // The file open beside the thread, if any. Held here rather than in a message
  // because the panel is a sibling of the whole conversation — and because
  // opening a second file has to replace the first, which it cannot do if each
  // turn owns its own.
  const [viewing, setViewing] = React.useState<Attachment | null>(null)
  const scrollRef = React.useRef<HTMLDivElement>(null)
  // Whether to keep the thread pinned to the bottom. Driven by the user's own
  // wheel/touch gestures (see trackStickIntent) so our programmatic scrolls
  // never fight them, and the last measured height so we only follow real
  // growth rather than every token mutation.
  const stickRef = React.useRef(true)
  const lastHeightRef = React.useRef(0)

  const lastMessage = messages[messages.length - 1]

  // The sidebar's count is the last *persisted* total — it doesn't move until
  // this turn is saved server-side, so a turn in flight is added on top from
  // what's been tallied client-side as it streams (see use-chat.ts). Neither
  // half is server-verified against the tokenizer, so this is an estimate for
  // the composer's status row, not an accounting figure.
  //
  // `prunedTokens` is subtracted rather than the total swapped for a smaller
  // one: `chatSummary.tokens` is also what the usage page bills from, and has
  // to keep counting every token this chat ever cost regardless of what a
  // later `/prune` excludes from the model's own view of it — see
  // `lib/context-floor.ts`.
  const contextTokensUsed =
    Math.max(0, (chatSummary?.tokens ?? 0) - prunedTokens) +
    (isSending ? (lastMessage?.tokenCount ?? 0) : 0)

  const showIdle = !isLoading && !isSending && lastMessage?.role === "assistant"
  // Waiting in line is not the same as generating: no tokens are coming yet, and
  // the reason is the queue rather than a slow model. It gets its own state so the
  // thread can say which, instead of showing a "generating" mark for a turn that
  // hasn't reached the GPU.
  const waiting = isSending && queuedAhead !== null && queuedAhead > 0
  const streaming = isSending && !waiting && lastMessage?.role === "assistant"
  // Show "Thinking…" whenever the model is *currently* inside a <|thinking|>
  // block (its content is hidden), or before any answer text has appeared at
  // all (the initial wait). Otherwise it's actively producing the answer, so
  // show "Generating…". Tracking the live block state — rather than latching on
  // "the answer has started" — means mid-reply reasoning is reflected too.
  //
  // A call in flight outranks both. It is the one phase where the wait is not
  // the model's at all — nothing is being generated while a search runs — so
  // saying "Generating…" over it would be a plain lie.
  //
  // The call itself, not just the fact of one: the row names the tool and shows
  // what it was asked, the same way the thinking state names what it is doing.
  const call = streaming
    ? callInFlight(lastMessage.working, lastMessage.toolAnswers ?? [])
    : null
  const searching = call !== null
  const inThinking = streaming && isInThinkingBlock(lastMessage.content)
  const showThinking =
    streaming &&
    !searching &&
    (inThinking || !hasVisibleContent(lastMessage.content))
  const showGenerating = streaming && !searching && !showThinking

  // Whether anything is written to the right of the rule. Generating says
  // nothing — the tokens arriving are the message itself, which the reader can
  // already see — so the rule would otherwise trail off into blank row.
  const label = searching || waiting || showThinking

  // One figure, and it is everything this conversation has cost: the context so
  // far plus what this turn has written into it, which is what
  // `contextTokensUsed` already is. It only ever climbs.
  //
  // It used to be whichever half matched the phase — the context while the
  // question went out, the reply's own count once tokens came back — and the
  // changeover dropped the number to zero mid-turn and started again. Two
  // honest figures, and the sequence between them a lie: nothing about that
  // moment resets anything.
  //
  // So the arrow carries the phase instead, and it is the only thing that
  // changes hands. Queuing, a tool call, and the beat before the first token
  // are the model being *given* something; from the first token on it is
  // writing. A context of zero is a chat whose summary has not loaded rather
  // than a turn that sent nothing, so it shows no figure at all.
  const written = isSending ? (lastMessage?.tokenCount ?? 0) : 0
  const statusFlow: TokenFlowProps | null =
    contextTokensUsed > 0 && !showIdle
      ? {
          direction: waiting || searching || written === 0 ? "in" : "out",
          count: contextTokensUsed,
        }
      : null

  // A user gesture near the bottom (re-)engages auto-follow; scrolling up to
  // read disengages it. Reading live scrollTop here — not during our own
  // programmatic scroll — is what keeps the two from fighting.
  const trackStickIntent = React.useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    stickRef.current = distanceFromBottom < STICK_TO_BOTTOM_THRESHOLD_PX
  }, [])

  // New chat: snap back to following, and forget the previous thread's height
  // so the first render of this one counts as growth (see below).
  React.useEffect(() => {
    stickRef.current = true
    lastHeightRef.current = 0
  }, [chatId])

  // A file belongs to the conversation it was sent in, so it closes with it.
  React.useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setViewing(null)
  }, [chatId])

  // Follow the bottom as content arrives — but only on genuine growth (a new
  // line, not an in-place token edit) and only while following.
  //
  // Always instant, never `behavior: "smooth"`. Streaming used to glide, and a
  // fast model breaks exactly that: a line wrapping is a new `scrollTo` call,
  // a fast enough model wraps a line every render, and each call retargets an
  // animation that hadn't finished chasing the last target — so the visible
  // position permanently trails the true bottom instead of catching up, and
  // the newest text sits behind the composer until generation pauses long
  // enough for the animation to land. Snapping every growth event keeps the
  // view exactly pinned regardless of how fast content arrives; the growth
  // events are themselves small (one wrapped line), which is what keeps this
  // reading as a glide rather than a jump.
  React.useEffect(() => {
    const el = scrollRef.current
    if (!el || !stickRef.current) return
    const grew = el.scrollHeight > lastHeightRef.current
    lastHeightRef.current = el.scrollHeight
    if (!grew) return
    el.scrollTo({ top: el.scrollHeight, behavior: "auto" })
  }, [messages, isSending])

  return (
    // A row, so opening a file narrows the thread rather than covering it. The
    // conversation column is centred in whatever width is left, which is what
    // makes it drift left as the panel comes in — no second layout, and nothing
    // to put back when it closes.
    <div className="flex flex-1 overflow-hidden">
      <div
        className={cn(
          "flex min-w-0 flex-1 flex-col overflow-hidden",
          // There is not room for both on a narrow screen, and half a
          // conversation beside half a document is neither. The panel wins,
          // because it is the thing just asked for.
          viewing && "hidden lg:flex"
        )}
      >
        {/* A flex column, so the thread below can be told to absorb whatever
          height the messages don't use. Without that, `sticky bottom-0` on the
          composer does nothing until the thread is long enough to scroll —
          sticky only pins an element that would otherwise leave the viewport —
          and a two-message conversation left the composer stranded halfway up
          the page with empty space under it. */}
        <div
          ref={scrollRef}
          onWheel={trackStickIntent}
          onTouchMove={trackStickIntent}
          className="flex flex-1 flex-col overflow-y-auto"
        >
          {/* `flex-1` is the piece that puts the composer on the floor: short
            threads stretch this block to fill the column and push the composer
            down, long ones grow past it and scroll as before (a flex item's
            `min-height: auto` is what stops the `flex-basis: 0` from crushing
            it). Messages stay top-aligned either way — a two-line exchange
            floating at the bottom of an empty page reads as the end of
            something rather than the start.

            gap-7 rather than gap-4: replies are now set at 16px with a 1.65
            line, and at that size four units between turns reads as one
            continuous column of text instead of an exchange. The air is what
            separates who is speaking, since the assistant side has no bubble to
            do it. */}
          <div className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-7 px-4 py-8">
            {isLoading ? (
              <div className="flex justify-center px-1 py-8">
                <Throbber mode="loading" />
              </div>
            ) : (
              messages.map((message, i) => {
                const id = message.id
                return (
                  <ChatMessageBubble
                    key={i}
                    message={message}
                    streaming={
                      isSending &&
                      i === messages.length - 1 &&
                      message.role === "assistant"
                    }
                    // Only assistant replies that exist server-side are rateable —
                    // there is nothing to attach a verdict to otherwise.
                    onRate={
                      id && message.role === "assistant"
                        ? (feedback) => void rateMessage(id, feedback)
                        : undefined
                    }
                    // Both re-ask rather than rewrite — see `UserActions`. Edit
                    // routes through the composer so the question can be changed,
                    // and so it is sent by the one path that owns sending.
                    onEdit={(content) => setDraft({ text: content })}
                    onResend={(content) =>
                      void sendMessage(content, model, budget)
                    }
                    // No model chosen yet means the list is still loading, and a
                    // send with an empty id would fail on the socket rather than
                    // in the UI.
                    busy={isSending || !model}
                    onOpenAttachment={setViewing}
                  />
                )
              })
            )}

            {/* Sent while the turn above is still streaming, not yet part of
              it — see `queued-message.tsx`. Below every real message and above
              the status row, the same place a reply-in-progress's own next
              word would land. */}
            {queued.map((message) => (
              <QueuedMessage key={message.id} text={message.content} />
            ))}

            {/* One fixed-height slot for all throbber states, so the
              thinking → generating → idle swaps don't reflow the thread. */}
            {(showThinking ||
              showGenerating ||
              searching ||
              showIdle ||
              waiting) && (
              // Mark, then the count, then what is being done. One
              // Throbber instance rather than one per state — that's what lets
              // thinking → generating → idle morph instead of swapping, since the
              // element is never unmounted.
              //
              // items-center, not items-baseline: the mark is a ball, so its
              // optical centre is its middle rather than its bottom edge, and a
              // baseline row would hang it below the line of the label.
              <div className="flex items-center gap-3 px-1 text-sm">
                <Throbber
                  stream={tokenArt}
                  mode={
                    waiting
                      ? "loading"
                      : searching
                        ? "tool"
                        : showThinking
                          ? "thinking"
                          : showGenerating
                            ? "generating"
                            : "idle"
                  }
                />
                {/* Time, then size, then what is being done — the reading
                  order of the row, and each rule drawn only when there is
                  something on both sides of it. The count sits here rather than
                  filed at the row's right edge, where it used to be, because it
                  is the subject of the row and not a measurement of it.

                  A finished turn keeps the clock and drops everything else: the
                  count it ended on is the conversation's own total, which the
                  composer under it already carries. */}
                {isSending && (
                  <TurnClock
                    // Keyed to the turn, so a second question in the same
                    // conversation starts a new clock rather than reading the
                    // first one's.
                    key={`${chatId}:${messages.length}`}
                  />
                )}
                {showIdle && lastMessage?.elapsedSecs != null && (
                  <span className="shrink-0 text-xs text-muted-foreground">
                    {workedFor(lastMessage.elapsedSecs)}
                  </span>
                )}
                {isSending && statusFlow && RULE}
                {statusFlow && <TokenFlow {...statusFlow} />}
                {statusFlow && label && RULE}
                {/* A fixed word rather than the rotating ThinkingLabel: that one
                  guesses at what a model might be doing, and here we know —
                  down to which tool and what it was handed. */}
                {call && <ToolLabel call={call} />}
                {waiting && (
                  <span className="text-muted-foreground">
                    Queued — {queuedAhead} {queuedAhead === 1 ? "job" : "jobs"}{" "}
                    ahead on the GPU. You can close this tab; the reply will be
                    here.
                  </span>
                )}
                {showThinking && <ThinkingLabel />}
                {/* Takes the slack, so a long tool argument truncates against
                  it rather than stretching the row. */}
                <span className="min-w-0 flex-1" />
              </div>
            )}

            {error && <LoadError title={error} />}
          </div>

          {/* Sticky, and *inside* the scroller rather than a sibling below it, so
            the thread passes behind the composer as you scroll back through it
            instead of being clipped at a hard edge. Sticky rather than absolute
            on purpose: the composer still takes real space at the end of the
            flow, so the last reply clears it at the bottom of the thread without
            anyone maintaining a padding constant against a box whose height
            changes with attachments and wrapped lines. */}
          <div className="sticky bottom-0 z-10 mx-auto w-full max-w-2xl px-4 pb-4">
            <ChatComposer
              key={chatId}
              plan={localPlan.plan}
              onPlanChange={localPlan.save}
              onSend={sendMessage}
              sending={isSending}
              onCancel={cancel}
              title={chatSummary?.title ?? "New chat"}
              onRenameTitle={(next) => void renameChat(chatId, next)}
              contextTokensUsed={contextTokensUsed}
              unseenTokens={prunedTokens}
              draft={draft}
              onPruneContext={() => pruneContext(model)}
            />
          </div>
        </div>
      </div>

      {viewing && (
        <AttachmentViewer
          // Keyed by the file, so opening a second one remounts rather than
          // showing the first one's contents under the second one's name while
          // the fetch is in flight.
          key={viewing.id}
          attachment={viewing}
          onClose={() => setViewing(null)}
        />
      )}
    </div>
  )
}
