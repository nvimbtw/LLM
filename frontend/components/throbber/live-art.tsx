"use client"

import * as React from "react"
import { usePrefersReducedMotion } from "@/hooks/use-reduced-motion"
import { ArtStream } from "@/lib/live-throbber/stream"
import { animateArt } from "@/lib/live-throbber/canvas"
import { cn } from "@/lib/utils"
import {
  TrellisThrobber,
  type ThrobberProps,
} from "@/components/ui/trellis-throbber"

export type ArtStatus = "starting" | "wgpu" | "fallback"
const STARTUP_TIMEOUT_MS = 8000

export interface LiveArtProps extends ThrobberProps {
  stream?: ArtStream
  /** Enlarged only by the fixture; production uses --size-live-throbber. */
  size?: number
  onStatus?: (status: ArtStatus) => void
}

export function LiveArt({
  mode = "loading",
  stream,
  size,
  className,
  onStatus,
  ...colours
}: LiveArtProps) {
  const [ownStream] = React.useState(() => new ArtStream())
  const source = stream ?? ownStream
  const reduced = usePrefersReducedMotion()
  const canvas = React.useRef<HTMLCanvasElement>(null)
  const modeRef = React.useRef(mode)
  const [status, setStatus] = React.useState<ArtStatus>("starting")

  React.useEffect(() => {
    modeRef.current = mode
    // A resting canvas may have stopped scheduling frames.
    source.wake()
  }, [mode, source])

  React.useEffect(() => {
    onStatus?.(status)
  }, [status, onStatus])

  React.useEffect(() => {
    if (!canvas.current) return
    let cancelled = false
    const controller = new AbortController()
    let stop: (() => void) | undefined
    const failed = () => {
      clearTimeout(deadline)
      if (!cancelled) setStatus("fallback")
    }
    const deadline = setTimeout(() => {
      failed()
      cancelled = true
      controller.abort()
      stop?.()
    }, STARTUP_TIMEOUT_MS)
    void animateArt({
      canvas: canvas.current,
      stream: source,
      reduced,
      signal: controller.signal,
      mode: () => modeRef.current,
      ready: () => {
        clearTimeout(deadline)
        if (!cancelled) setStatus("wgpu")
      },
      failed,
    })
      .then((cleanup) => {
        // Offscreen canvases wait to draw; only initialization has a deadline.
        clearTimeout(deadline)
        if (cancelled) cleanup()
        else stop = cleanup
      })
      .catch(failed)
    return () => {
      cancelled = true
      controller.abort()
      clearTimeout(deadline)
      stop?.()
    }
  }, [source, reduced, size])

  return (
    <span
      role="status"
      aria-label={
        mode === "idle" ? "Ready" : mode.charAt(0).toUpperCase() + mode.slice(1)
      }
      data-art-status={status}
      className={cn(
        "relative inline-flex shrink-0 items-center justify-center",
        className
      )}
      style={{
        width: size ?? "var(--size-live-throbber)",
        height: size ?? "var(--size-live-throbber)",
      }}
    >
      {status !== "wgpu" && (
        <span aria-hidden>
          <TrellisThrobber mode={mode} {...colours} />
        </span>
      )}
      <canvas
        ref={canvas}
        aria-hidden
        className={cn(
          "absolute block size-full rounded-full",
          status === "wgpu" ? "opacity-100" : "opacity-0"
        )}
      />
    </span>
  )
}
