"use client"

import { useLayoutEffect, useRef } from "react"

/** Measures the composer's natural body height and updates its shell so CSS can animate wrapping, upload changes and clearing. Owns and disconnects its resize observer. */
export function useComposerHeight() {
  const shellRef = useRef<HTMLDivElement>(null)
  const bodyRef = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const shell = shellRef.current
    const body = bodyRef.current
    if (!shell || !body) return
    const style = getComputedStyle(shell)
    const borderHeight =
      parseFloat(style.borderTopWidth) + parseFloat(style.borderBottomWidth)
    /** Sets the shell's target height from the natural content and its fixed border, without introducing editor state. */
    const measure = () => {
      if (!body.getClientRects().length) return
      shell.style.height = `${body.getBoundingClientRect().height + borderHeight}px`
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(body)
    return () => observer.disconnect()
  }, [])
  return { shellRef, bodyRef }
}
