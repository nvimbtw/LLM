"use client"

import * as React from "react"

import { cn } from "@/lib/utils"
import { useAudioLevel, type AudioBands } from "@/hooks/use-audio-level"
import { usePrefersReducedMotion } from "@/hooks/use-reduced-motion"

export interface VoiceWaveformProps {
  /** Only ever said out loud to a screen reader — where the audio goes is not
   *  something a person can see, and this is the one place it is stated. The
   *  microphone's own tooltip carries it for everyone else. See
   *  `docs/decisions/voice-input-uses-the-browser.md`. */
  onDevice?: boolean
  className?: string
}

/** How short a bar gets in silence. Not zero: three bars that vanish read as a
 *  broken control rather than a quiet room. */
const MIN_EXTENT = 0.2

/** Left to right, in band order — low, mid, high. Three of them because two
 *  read as a loading state and four stop being legible at this size. */
const BAR_COUNT = 3

/**
 * Three bars that rise and fall with the voice.
 *
 * Mounted only while dictating — mounting is what opens the microphone and
 * unmounting is what closes it, so there is no state here saying whether it is
 * running. The bars are written to imperatively from the audio callback; a
 * render per frame to set three transforms would be the most expensive thing on
 * the page.
 */
export function VoiceWaveform({ onDevice, className }: VoiceWaveformProps) {
  const bars = React.useRef<(HTMLSpanElement | null)[]>([])
  const reducedMotion = usePrefersReducedMotion()

  const paint = React.useCallback((bands: AudioBands) => {
    for (let i = 0; i < bands.length; i++) {
      const bar = bars.current[i]
      if (!bar) continue
      bar.style.transform = `scaleY(${MIN_EXTENT + (1 - MIN_EXTENT) * bands[i]})`
    }
  }, [])

  const error = useAudioLevel(!reducedMotion, paint)

  return (
    <span
      className={cn(
        "flex h-3 items-center gap-1",
        // Nothing to draw: the bars stay at rest and step back rather than
        // sitting there at full strength implying a level they are not reading.
        error && "opacity-40",
        className
      )}
    >
      <span className="sr-only" role="status">
        {onDevice ? "Listening — on device" : "Listening"}
      </span>
      {Array.from({ length: BAR_COUNT }, (_, i) => (
        <span
          key={i}
          aria-hidden="true"
          ref={(el) => {
            bars.current[i] = el
          }}
          className="block h-full w-0.5 origin-center rounded-full bg-primary/70"
          // The resting shape, and the shape it keeps under reduced motion.
          style={{ transform: `scaleY(${MIN_EXTENT})` }}
        />
      ))}
    </span>
  )
}
