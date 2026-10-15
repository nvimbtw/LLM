"use client"

import { useCallback, useEffect, useRef, useState, type RefObject } from "react"

const MESSAGE_INSET_PX = 24
const NAVIGATION_MIN_MS = 240
const NAVIGATION_MAX_MS = 420
const NAVIGATION_PX_PER_MS = 12

/** Eases a question jump over 240–420ms, or moves immediately with reduced motion. Owns and releases its frame and gesture listeners on completion, interruption or caller cleanup. */
function animateQuestionNavigation(
  viewport: HTMLDivElement,
  targetScrollTop: number
): () => void {
  const startTop = viewport.scrollTop
  const targetTop = Math.max(
    0,
    Math.min(targetScrollTop, viewport.scrollHeight - viewport.clientHeight)
  )
  const distance = targetTop - startTop
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)")
  if (reducedMotion.matches || Math.abs(distance) < 1) {
    viewport.scrollTo({ top: targetTop, behavior: "instant" })
    return () => {}
  }
  const duration = Math.min(
    NAVIGATION_MAX_MS,
    NAVIGATION_MIN_MS + Math.abs(distance) / NAVIGATION_PX_PER_MS
  )
  const startedAt = performance.now()
  let frame: number | null = null
  /** Stops this jump without changing the reader's position and releases its gesture listeners. */
  const cancel = () => {
    if (frame !== null) cancelAnimationFrame(frame)
    frame = null
    for (const event of ["wheel", "touchstart", "pointerdown", "keydown"])
      viewport.removeEventListener(event, cancel)
  }
  /** Applies one eased position, honoring reduced-motion changes before scheduling the next frame. */
  const step = (now: number) => {
    const progress = reducedMotion.matches
      ? 1
      : Math.min(1, (now - startedAt) / duration)
    const eased = progress * progress * (3 - 2 * progress)
    viewport.scrollTo({ top: startTop + distance * eased, behavior: "instant" })
    if (progress === 1) cancel()
    else frame = requestAnimationFrame(step)
  }
  for (const event of ["wheel", "touchstart", "pointerdown", "keydown"])
    viewport.addEventListener(event, cancel, { passive: true })
  frame = requestAnimationFrame(step)
  return cancel
}

/** Returns the reading height above the measured composer inset, while the full transcript remains visible behind its glass. */
function readingHeight(viewport: HTMLDivElement): number {
  return Math.max(
    0,
    viewport.clientHeight - parseFloat(getComputedStyle(viewport).paddingBottom)
  )
}

/** Finds the question whose center has reached the viewport center; at either end, the first or latest question is current. */
function messageAtReadingLine(
  viewport: HTMLDivElement,
  messages: HTMLElement[]
): number {
  if (!messages.length) return 0
  if (viewport.scrollTop <= 1) return 0
  if (viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight <= 1)
    return messages.length - 1
  const line =
    viewport.getBoundingClientRect().top + readingHeight(viewport) / 2 + 1
  let low = 0
  let high = messages.length
  while (low < high) {
    const middle = Math.floor((low + high) / 2)
    const bounds = messages[middle].getBoundingClientRect()
    if (bounds.top + bounds.height / 2 <= line) low = middle + 1
    else high = middle
  }
  return Math.max(0, low - 1)
}

/** Tracks scroll and layout changes at most once per frame, releasing its observers, listener and pending frame on cleanup. */
function observeChatPosition(
  viewport: HTMLDivElement,
  content: HTMLDivElement,
  onPosition: (index: number) => void
): () => void {
  const messages = Array.from(
    content.querySelectorAll<HTMLElement>("[data-chat-question]")
  )
  let frame: number | null = null
  /** Schedules one read after scrolling, wrapping, composer resizing or trace expansion. */
  const schedule = () => {
    if (frame !== null) return
    frame = requestAnimationFrame(() => {
      frame = null
      onPosition(messageAtReadingLine(viewport, messages))
    })
  }
  const observer = new ResizeObserver(schedule)
  observer.observe(viewport)
  observer.observe(content)
  viewport.addEventListener("scroll", schedule, { passive: true })
  schedule()
  return () => {
    observer.disconnect()
    viewport.removeEventListener("scroll", schedule)
    if (frame !== null) cancelAnimationFrame(frame)
  }
}

/** Owns the current question and cancellable eased navigation within one conversation's scroll container. */
export function useChatPosition(
  scrollRef: RefObject<HTMLDivElement | null>,
  contentRef: RefObject<HTMLDivElement | null>,
  messageCount: number
) {
  const [activeIndex, setActiveIndex] = useState(0)
  const cancelNavigationRef = useRef<(() => void) | null>(null)
  useEffect(() => () => cancelNavigationRef.current?.(), [scrollRef])
  useEffect(() => {
    const viewport = scrollRef.current
    const content = contentRef.current
    if (!viewport || !content) return
    return observeChatPosition(viewport, content, setActiveIndex)
  }, [scrollRef, contentRef, messageCount])

  /** Centers a selected question without moving the surrounding page; questions taller than the viewport align near its top. */
  const jumpToMessage = useCallback(
    (index: number) => {
      const viewport = scrollRef.current
      const message = contentRef.current?.querySelector<HTMLElement>(
        `[data-message-index="${index}"]`
      )
      if (!viewport || !message) return
      cancelNavigationRef.current?.()
      const offset =
        message.getBoundingClientRect().top -
        viewport.getBoundingClientRect().top
      const inset = Math.max(
        MESSAGE_INSET_PX,
        (readingHeight(viewport) - message.getBoundingClientRect().height) / 2
      )
      cancelNavigationRef.current = animateQuestionNavigation(
        viewport,
        viewport.scrollTop + offset - inset
      )
    },
    [scrollRef, contentRef]
  )
  return {
    activeIndex: Math.min(activeIndex, Math.max(0, messageCount - 1)),
    jumpToMessage,
  }
}
