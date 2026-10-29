"use client"

import { memo, useCallback, useEffect, useMemo, useRef, type RefObject } from "react"
import type { ChatMessage } from "@/lib/chat-messages"
import { useChatPosition } from "@/hooks/use-chat-position"
import "./chat-position-rail.css"

export interface ChatPositionRailProps {
  messages: ChatMessage[]
  scrollRef: RefObject<HTMLDivElement | null>
  contentRef: RefObject<HTMLDivElement | null>
  onNavigate: () => void
}

/** Produces bounded plain text for exchange previews and accessible destinations. */
function excerpt(text: string): string {
  return text
    .slice(0, 1200)
    .replace(/<\|[^>]*\|>/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 400)
}

/**
 * One tick in the rail.
 *
 * The rail is a child of the thread, and the thread redraws while a reply
 * streams, so every tick used to be rebuilt — label normalisation and button
 * included — on every frame, for questions that had not changed. A tick only
 * depends on its own message, its index, whether it is current, and a stable
 * jump callback, so all of those bailing out is the normal case.
 */
const PositionTick = memo(function PositionTick({
  message,
  index,
  messageIndex,
  active,
  onJump,
}: {
  message: ChatMessage
  index: number
  messageIndex: number
  active: boolean
  onJump: (messageIndex: number) => void
}) {
  return (
    <button
      type="button"
      aria-label={`Question ${index + 1}: ${excerpt(message.content)}`}
      aria-current={active ? "location" : undefined}
      className="fluid-chat-tick"
      onClick={() => onJump(messageIndex)}
    >
      <span aria-hidden="true" />
    </button>
  )
})

/** Presents fluid exchange strokes and neutral previews; proximity changes DOM styles without rerendering the conversation. */
export function ChatPositionRail({
  messages,
  scrollRef,
  contentRef,
  onNavigate,
}: ChatPositionRailProps) {
  const questions = useMemo(
    () =>
      messages.flatMap((message, messageIndex) =>
        message.role === "user" ? [{ message, messageIndex }] : []
      ),
    [messages]
  )
  const { activeIndex, jumpToMessage } = useChatPosition(
    scrollRef,
    contentRef,
    questions.length
  )
  /** Stable for the life of a conversation, so a tick is skipped when nothing about it changed. */
  const jump = useCallback(
    (messageIndex: number) => {
      onNavigate()
      jumpToMessage(messageIndex)
    },
    [onNavigate, jumpToMessage]
  )
  const railRef = useRef<HTMLElement>(null)
  const previewRef = useRef<HTMLDivElement>(null)
  const messagesRef = useRef(messages)
  const questionsRef = useRef(questions)
  useEffect(() => {
    messagesRef.current = messages
    questionsRef.current = questions
  }, [messages, questions])
  const activeRef = useRef(activeIndex)
  useEffect(() => {
    activeRef.current = activeIndex
  }, [activeIndex])
  useEffect(() => {
    const rail = railRef.current
    const preview = previewRef.current
    if (!rail || !preview) return
    const buttons = Array.from(
      rail.querySelectorAll<HTMLButtonElement>("button")
    )
    let selected: number | null = null
    /** Paints proximity and places the adjacent preview within the viewport. */
    const paint = (index: number | null) => {
      const affected = new Set<number>([activeRef.current])
      for (const center of [selected, index]) {
        if (center === null) continue
        for (
          let i = Math.max(0, center - 8);
          i <= Math.min(buttons.length - 1, center + 8);
          i++
        )
          affected.add(i)
      }
      selected = index
      affected.forEach((i) => {
        const button = buttons[i]
        if (!button) return
        const weight =
          index === null ? 0 : Math.exp(-Math.pow((i - index) / 2, 2))
        button.style.setProperty(
          "--stroke-width",
          `${(i === activeRef.current ? 12 : 7) + 21 * weight}px`
        )
        button.style.setProperty(
          "--stroke-opacity",
          `${Math.min(1, (i === activeRef.current ? 0.9 : 0.3) + 0.45 * weight)}`
        )
      })
      preview.hidden = index === null
      if (index === null) return
      const question = questionsRef.current[index]
      const latestMessages = messagesRef.current
      const nextQuestion =
        questionsRef.current[index + 1]?.messageIndex ?? latestMessages.length
      const reply = latestMessages
        .slice(question.messageIndex + 1, nextQuestion)
        .find((message) => message.role === "assistant")
      const promptNode = preview.querySelector("[data-preview-prompt]")
      const replyNode = preview.querySelector("[data-preview-reply]")
      if (promptNode)
        promptNode.textContent =
          excerpt(question.message.content) || "Attachment"
      if (replyNode)
        replyNode.textContent = reply ? excerpt(reply.content) : "No reply yet"
      const bounds = buttons[index].getBoundingClientRect()
      const parent = preview.parentElement!.getBoundingClientRect()
      const top = Math.max(
        12,
        Math.min(
          window.innerHeight - preview.offsetHeight - 12,
          bounds.top + bounds.height / 2 - preview.offsetHeight / 2
        )
      )
      preview.style.top = `${top - parent.top}px`
    }
    /** Chooses the nearest stroke from the pointer's vertical position. */
    const point = (event: PointerEvent) => {
      const index = Math.max(
        0,
        Math.min(
          buttons.length - 1,
          Math.floor(
            (event.clientY -
              rail.getBoundingClientRect().top +
              rail.scrollTop) /
              11
          )
        )
      )
      paint(index)
    }
    /** Restores resting strokes after pointer exit or keyboard focus leaves. */
    const leave = () => {
      paint(null)
    }
    /** Reveals the focused exchange using the same preview as the pointer. */
    const focus = (event: FocusEvent) => {
      const index = buttons.indexOf(event.target as HTMLButtonElement)
      if (index >= 0) paint(index)
    }
    /** Moves keyboard focus sequentially and dismisses the preview with Escape. */
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault()
        paint(null)
        return
      }
      const direction =
        event.key === "ArrowUp" ? -1 : event.key === "ArrowDown" ? 1 : 0
      const index = buttons.indexOf(event.target as HTMLButtonElement)
      if (!direction || index < 0) return
      event.preventDefault()
      buttons[
        Math.max(0, Math.min(buttons.length - 1, index + direction))
      ]?.focus()
    }
    /** Repositions the open preview after scrolling or viewport resizing. */
    const reposition = () => {
      if (selected !== null) paint(selected)
    }
    paint(null)
    rail.addEventListener("pointermove", point)
    rail.addEventListener("pointerleave", leave)
    rail.addEventListener("focusin", focus)
    /** Closes the preview only when focus leaves the rail, rather than between adjacent strokes. */
    const blur = (event: FocusEvent) => {
      if (!rail.contains(event.relatedTarget as Node | null)) paint(null)
    }
    rail.addEventListener("focusout", blur)
    rail.addEventListener("keydown", key)
    rail.addEventListener("scroll", reposition)
    window.addEventListener("resize", reposition)
    return () => {
      rail.removeEventListener("pointermove", point)
      rail.removeEventListener("pointerleave", leave)
      rail.removeEventListener("focusin", focus)
      rail.removeEventListener("focusout", blur)
      rail.removeEventListener("keydown", key)
      rail.removeEventListener("scroll", reposition)
      window.removeEventListener("resize", reposition)
    }
  }, [questions.length])
  useEffect(() => {
    const rail = railRef.current
    const current = rail?.querySelector<HTMLElement>(
      "[aria-current='location']"
    )
    if (!rail || !current) return
    rail
      .querySelectorAll<HTMLButtonElement>("button")
      .forEach((button, index) => {
        button.style.setProperty(
          "--stroke-width",
          `${index === activeIndex ? 12 : 7}px`
        )
        button.style.setProperty(
          "--stroke-opacity",
          index === activeIndex ? ".9" : ".3"
        )
      })
    rail.scrollTo({
      top: current.offsetTop - (rail.clientHeight - current.clientHeight) / 2,
      behavior: "instant",
    })
  }, [activeIndex, questions.length])
  if (!questions.length) return null
  return (
    <div className="fluid-chat-position pointer-events-none absolute top-1/2 left-6 z-20 hidden -translate-y-1/2 @3xl/chat:block">
      <nav
        ref={railRef}
        aria-label="Conversation questions"
        className="fluid-chat-index pointer-events-auto"
      >
        {questions.map(({ message, messageIndex }, index) => (
          <PositionTick
            key={messageIndex}
            message={message}
            index={index}
            messageIndex={messageIndex}
            active={index === activeIndex}
            onJump={jump}
          />
        ))}
      </nav>
      <div
        ref={previewRef}
        hidden
        className="fluid-chat-preview"
        aria-hidden="true"
      >
        <p data-preview-prompt className="line-clamp-1 font-medium" />
        <p
          data-preview-reply
          className="mt-1.5 line-clamp-3 text-muted-foreground"
        />
      </div>
    </div>
  )
}
