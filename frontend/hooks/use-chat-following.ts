"use client"

import { useCallback, useLayoutEffect, useRef, type RefObject } from "react"

const STICK_TO_BOTTOM_THRESHOLD_PX = 120

export interface UseChatFollowingOptions {
  scrollRef: RefObject<HTMLDivElement | null>
  contentRef: RefObject<HTMLDivElement | null>
  composerRef: RefObject<HTMLDivElement | null>
}

/** Reserves the measured composer's height within its transcript and follows growth only while pinned. The three refs belong to one chat pane; cleanup releases the inset, observers and gesture frame. */
export function useChatFollowing({
  scrollRef,
  contentRef,
  composerRef,
}: UseChatFollowingOptions) {
  const followingRef = useRef(true)
  const gestureFrameRef = useRef<number | null>(null)

  /** Cancels a pending gesture read and pauses following before message navigation. */
  const pauseFollowing = useCallback(() => {
    followingRef.current = false
    if (gestureFrameRef.current !== null) {
      cancelAnimationFrame(gestureFrameRef.current)
      gestureFrameRef.current = null
    }
  }, [])

  /** Pauses while a wheel/touch gesture moves the viewport, then resumes if its resulting position is near the bottom. */
  const trackStickIntent = useCallback(() => {
    pauseFollowing()
    gestureFrameRef.current = requestAnimationFrame(() => {
      gestureFrameRef.current = null
      const viewport = scrollRef.current
      if (!viewport) return
      const distance =
        viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight
      followingRef.current = distance < STICK_TO_BOTTOM_THRESHOLD_PX
    })
  }, [pauseFollowing, scrollRef])

  useLayoutEffect(() => {
    const viewport = scrollRef.current
    const content = contentRef.current
    const composer = composerRef.current
    if (!viewport || !content || !composer) return
    /** Reserves each animation frame's composer height beneath the latest content and follows instantly unless the reader paused. */
    const follow = () => {
      if (
        !viewport.getClientRects().length ||
        !composer.getClientRects().length
      )
        return
      const inset = `${composer.getBoundingClientRect().height}px`
      if (viewport.style.getPropertyValue("--chat-composer-height") !== inset)
        viewport.style.setProperty("--chat-composer-height", inset)
      if (followingRef.current)
        viewport.scrollTo({ top: viewport.scrollHeight, behavior: "instant" })
    }
    const observer = new ResizeObserver(follow)
    observer.observe(viewport)
    observer.observe(content)
    observer.observe(composer)
    follow()
    return () => {
      observer.disconnect()
      viewport.style.removeProperty("--chat-composer-height")
      if (gestureFrameRef.current !== null)
        cancelAnimationFrame(gestureFrameRef.current)
      gestureFrameRef.current = null
    }
  }, [scrollRef, contentRef, composerRef])
  return { pauseFollowing, trackStickIntent }
}
