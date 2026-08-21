"use client"

import * as React from "react"
import { Mic, Square } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

export interface MicButtonProps {
  listening: boolean
  /** Whether the recogniser runs locally. Only ever surfaces in the tooltip:
   *  where the audio goes is not visible, and a person who wants to know should
   *  be able to find out from the control that starts it. */
  onDevice?: boolean
  onStart: () => void
  onStop: () => void
  disabled?: boolean
  className?: string
}

/** Longer than this and the press was a hold, so releasing ends dictation.
 *  Shorter and it was a click, which leaves it running. */
const HOLD_TO_TALK_MS = 350

/**
 * The microphone, next to Send. Two gestures on one control, told apart by how
 * long the press lasted: click to dictate until you click again, or hold while
 * you speak and let go — the shape of the gesture matches the length of the
 * thought, and neither has to be learned before the other works.
 */
export function MicButton({
  listening,
  onDevice,
  onStart,
  onStop,
  disabled,
  className,
}: MicButtonProps) {
  // Both facts about the press are read on release, and neither should cause a
  // render while a finger is down.
  const press = React.useRef<{ at: number; wasListening: boolean } | null>(null)

  const toggle = () => (listening ? onStop() : onStart())

  const handlePointerDown = (event: React.PointerEvent<HTMLButtonElement>) => {
    if (disabled) return
    // Keeps the caret in the composer: the spoken text is inserted at the
    // cursor, and a button that steals focus first has nowhere to put it.
    event.preventDefault()
    // Without capture a pointer that drifts off the button never delivers its
    // `pointerup`, and a hold-to-talk would never end.
    event.currentTarget.setPointerCapture(event.pointerId)
    press.current = { at: Date.now(), wasListening: listening }
    if (!listening) onStart()
  }

  const handlePointerUp = () => {
    const started = press.current
    press.current = null
    if (!started) return
    if (started.wasListening) onStop()
    else if (Date.now() - started.at >= HOLD_TO_TALK_MS) onStop()
  }

  return (
    <Button
      type="button"
      variant={listening ? "default" : "outline"}
      size="icon-sm"
      aria-label={listening ? "Stop dictating" : "Dictate"}
      aria-pressed={listening}
      title={
        onDevice
          ? "Dictate — recognised on this device"
          : "Dictate — audio goes to the browser's speech service"
      }
      disabled={disabled}
      className={cn("rounded-full", className)}
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      onKeyDown={(event) => {
        // The pointer path suppresses the click that would normally follow, so
        // the keyboard needs its own way in. Both gestures collapse to a plain
        // toggle here — there is no holding a key without repeat noise.
        if (event.key !== " " && event.key !== "Enter") return
        if (event.repeat) return
        event.preventDefault()
        toggle()
      }}
    >
      {/* The same swap the send button makes while a turn is in flight: filled
          blue, and a stop glyph, because stopping is what the control does now.
          The state is loud enough at that — it is the only filled circle in a
          row of outlined ones — and a pulse on top would say a second time what
          the colour already says once. What *moves* while dictation runs is the
          waveform in the drawer, and it moves because the voice does. */}
      {listening ? <Square className="fill-current" /> : <Mic />}
    </Button>
  )
}
