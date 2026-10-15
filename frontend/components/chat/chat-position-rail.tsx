"use client"

import { useEffect, useRef, type RefObject } from "react"

import type { ChatMessage } from "@/lib/chat-messages"
import { cn } from "@/lib/utils"
import { useChatPosition } from "@/hooks/use-chat-position"
import { Button } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

export interface ChatPositionRailProps {
  messages: ChatMessage[]
  scrollRef: RefObject<HTMLDivElement | null>
  contentRef: RefObject<HTMLDivElement | null>
  onNavigate: () => void
}

/** Labels a question with its ordinal and a bounded preview of the user's text. */
function messageLabel(message: ChatMessage, index: number): string {
  const preview = message.content
    .slice(0, 160)
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 120)
  return `Question ${index + 1}${preview ? ` — ${preview}` : ""}`
}

/** Renders an edge-anchored index of user questions, with keyboard-accessible previews and a longer stroke for the current exchange. */
export function ChatPositionRail({
  messages,
  scrollRef,
  contentRef,
  onNavigate,
}: ChatPositionRailProps) {
  const questions = messages.flatMap((message, messageIndex) =>
    message.role === "user" ? [{ message, messageIndex }] : []
  )
  const { activeIndex, jumpToMessage } = useChatPosition(
    scrollRef,
    contentRef,
    questions.length
  )
  const railRef = useRef<HTMLElement>(null)
  useEffect(() => {
    const rail = railRef.current
    const current = rail?.querySelector<HTMLElement>(
      "[aria-current='location']"
    )
    if (!rail || !current) return
    /** Keeps the current question in the bounded rail when selection or pane visibility changes, without moving the page. */
    const alignCurrent = () => {
      if (!rail.clientHeight) return
      rail.scrollTo({
        top: current.offsetTop - (rail.clientHeight - current.clientHeight) / 2,
        behavior: "instant",
      })
    }
    alignCurrent()
    const observer = new ResizeObserver(alignCurrent)
    observer.observe(rail)
    return () => observer.disconnect()
  }, [activeIndex, questions.length])
  if (!questions.length) return null

  return (
    <div className="pointer-events-none absolute top-1/2 left-6 z-20 hidden -translate-y-1/2 @3xl/chat:block">
      <nav
        ref={railRef}
        aria-label="Conversation questions"
        className="pointer-events-auto relative max-h-48 [scrollbar-width:none] overflow-y-auto overscroll-contain"
      >
        <TooltipProvider delay={250}>
          {questions.map(({ message, messageIndex }, index) => {
            const label = messageLabel(message, index)
            const active = index === activeIndex
            return (
              <Tooltip key={index}>
                <TooltipTrigger
                  render={
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      aria-label={label}
                      aria-current={active ? "location" : undefined}
                      className="group/position flex h-4 w-7 justify-start rounded-none pl-2 hover:bg-transparent focus-visible:ring-1 dark:hover:bg-transparent"
                      onClick={() => {
                        onNavigate()
                        jumpToMessage(messageIndex)
                      }}
                    >
                      <span
                        aria-hidden="true"
                        className={cn(
                          "block h-px transition-all duration-150 group-hover/position:bg-foreground group-focus-visible/position:bg-foreground motion-reduce:transition-none",
                          active
                            ? "w-4 bg-foreground/80"
                            : "w-2.5 bg-muted-foreground/60"
                        )}
                      />
                    </Button>
                  }
                />
                <TooltipContent side="right" className="max-w-64">
                  {label}
                </TooltipContent>
              </Tooltip>
            )
          })}
        </TooltipProvider>
      </nav>
    </div>
  )
}
