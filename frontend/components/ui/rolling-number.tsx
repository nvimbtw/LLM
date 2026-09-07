"use client"

import * as React from "react"

import { usePrefersReducedMotion } from "@/hooks/use-reduced-motion"
import { cn } from "@/lib/utils"

/**
 * A number that changes by rolling, one column per digit, the way a counter
 * with wheels in it does. For a figure that climbs while you watch — a token
 * count arriving over a socket — the movement is the reading: how fast the
 * column turns is how fast the tokens are coming, which no static number can
 * say.
 *
 * **Every column only ever rolls upward, including through the nine.** That is
 * the whole difficulty here. A column that shows the digit and nothing else has
 * to travel from 9 back down to 0, so a counter passing 19 → 20 has one wheel
 * going up and its neighbour dropping the long way down through every digit it
 * just came through. The fix is that a column is *two* stacked runs of 0–9 and
 * what it tracks is a row, not a digit: to reach 0 from 9 it steps to the
 * eleventh row, which is another 0. When it runs out of rows it drops back a
 * whole ten with the transition off — rows ten apart carry the same glyph, so
 * nothing moves on screen, and the column has room to keep climbing.
 *
 * Columns are keyed from the right, so a number growing a digit slides the new
 * one in at the left instead of renumbering every wheel under it.
 */
export interface RollingNumberProps {
  /** Counts upward. A value that falls still rolls up, the long way round. */
  value: number
  /**
   * Abbreviate past a thousand — `11.3k`, `2.4M`, `1.1B` — instead of grouping
   * every digit.
   *
   * It costs something, and the cost is the wheels: an abbreviated figure moves
   * on the hundred rather than on the unit, so a column that was turning
   * steadily goes still between steps. Worth it once a number is long enough
   * that its last digits are noise — six wheels of token count are five wheels
   * nobody reads — and not before.
   */
  compact?: boolean
  /**
   * Land the whole figure at once, near-instantly, instead of turning at the
   * pace a stream arrives at.
   *
   * The two are different events and should not look alike. A count of tokens
   * being written climbs a few at a time, and the wheels turning *is* the
   * stream; a count that jumps because something was handed over whole — a
   * message sent, a tool's answer coming back — has already happened by the
   * time it is drawn, and a leisurely roll would be animating a wait that
   * nobody is waiting on.
   */
  swift?: boolean
  className?: string
}

/** Where each abbreviation starts, largest first. `k` lowercase and the rest
 * capital is the ordinary convention, and `m` for a million would be the one
 * abbreviation here a reader could mistake for a smaller unit. */
const UNITS: { at: number; suffix: string }[] = [
  { at: 1e9, suffix: "B" },
  { at: 1e6, suffix: "M" },
  { at: 1e3, suffix: "k" },
]

/** Two runs of 0–9. The second is what a wheel rolls *into* to reach a smaller
 * digit without travelling backwards. */
const ROWS = 20

/** How long one column takes to reach its new row. Short: while a reply
 * streams, the ones column is asked for a new digit faster than this, and the
 * overlap is what turns discrete steps into a wheel that spins.
 *
 * Eased in and out (see the class on the strip) rather than out alone: a wheel
 * that leaves at full speed has already arrived by the time you look at it, and
 * only the settling is visible. Easing both ends is what makes it read as a
 * weight being turned. */
const ROLL_MS = 220

/** A figure that arrives whole, rather than climbing: fast enough to read as
 * the number having landed, slow enough that the wheels are seen to do it. */
const SWIFT_MS = 90

export function RollingNumber({
  value,
  compact = false,
  swift = false,
  className,
}: RollingNumberProps) {
  const reduced = usePrefersReducedMotion()
  // Formatted here rather than by the caller: the separators, the point and the
  // suffix have to be laid out beside the wheels, not printed into them.
  const text = format(Math.max(0, Math.round(value)), compact)
  const glyphs = [...text]

  return (
    <span className={cn("inline-flex items-center tabular-nums", className)}>
      {glyphs.map((glyph, i) => {
        const fromRight = glyphs.length - 1 - i
        return /\d/.test(glyph) ? (
          <Wheel
            key={`d${fromRight}`}
            digit={Number(glyph)}
            rolls={!reduced}
            ms={swift ? SWIFT_MS : ROLL_MS}
          />
        ) : (
          <span key={`s${fromRight}`} aria-hidden>
            {glyph}
          </span>
        )
      })}
    </span>
  )
}

function Wheel({
  digit,
  rolls,
  ms,
}: {
  digit: number
  rolls: boolean
  ms: number
}) {
  const strip = React.useRef<HTMLSpanElement>(null)
  // Which of the twenty rows is showing, or -1 before the first paint. Held in
  // a ref and written straight to the node: the row is not a function of the
  // digit — the same digit is two different rows depending on how the column
  // got there — so React cannot own it without re-rendering every wheel on
  // every token.
  const row = React.useRef(-1)

  React.useEffect(() => {
    const el = strip.current
    if (!el) return

    const roll = () => {
      const advance = (digit - (row.current % 10) + 10) % 10
      if (advance === 0) return

      let next = row.current + advance
      if (next >= ROWS) {
        jump(el, row.current - 10)
        next -= 10
      }
      row.current = next
      el.style.transform = offset(next)
    }

    if (row.current < 0) {
      // A wheel arrives showing zero and rolls to its digit on the next frame,
      // rather than mounting already on it. A figure that appears finished
      // never appears to move, and its arriving is the one thing about it worth
      // seeing — which is also what makes the count spin up when a message is
      // sent, where the row and the number both appear at the same instant.
      row.current = 0
      jump(el, 0)
      const frame = requestAnimationFrame(roll)
      return () => cancelAnimationFrame(frame)
    }

    roll()
  }, [digit])

  return (
    // The wheel is as wide as one digit and as tall as one line, and the run of
    // digits inside it is what moves. `aria-hidden` throughout: a screen reader
    // is given the number by the caller, and twenty digits read aloud is not a
    // number.
    <span
      aria-hidden
      // `1ch` is the width of a zero in whatever font this lands in, which for
      // tabular figures is the width of every digit — so the wheel is exactly
      // as wide as the glyph on it, with no measuring and no guessed em.
      className="relative inline-block h-[1.25em] w-[1ch] overflow-hidden"
    >
      <span
        ref={strip}
        className={cn(
          "absolute inset-x-0 top-0 flex flex-col",
          rolls && "transition-transform ease-in-out"
        )}
        style={{ transitionDuration: rolls ? `${ms}ms` : undefined }}
      >
        {Array.from({ length: ROWS }, (_, i) => (
          <span key={i} className="h-[1.25em] text-center leading-[1.25em]">
            {i % 10}
          </span>
        ))}
      </span>
    </span>
  )
}

/**
 * The digits to lay out. Truncated to the tenth rather than rounded: a counter
 * that reads 12.0k while the twelve thousandth token is still on its way is
 * claiming tokens that have not arrived, and this figure is watched precisely
 * while it climbs.
 */
function format(value: number, compact: boolean): string {
  if (compact) {
    for (const { at, suffix } of UNITS) {
      if (value < at) continue
      return `${(Math.floor((value / at) * 10) / 10).toFixed(1)}${suffix}`
    }
  }
  return value.toLocaleString("en-US")
}

/** A percentage of the strip's own height, so the wheel needs to know nothing
 * about the font it is set in. */
const offset = (row: number) => `translateY(-${(row * 100) / ROWS}%)`

/** Move without moving: used only between rows that carry the same glyph. */
function jump(el: HTMLElement, row: number) {
  // The `transition-property` longhand, not the shorthand: the shorthand would
  // also clear the duration React set inline, and clearing it silently drops
  // the column back to whatever the utility class happens to time at.
  el.style.transitionProperty = "none"
  el.style.transform = offset(row)
  // Read back, or the browser coalesces this write with the one after it and
  // animates the jump we just took pains to hide.
  void el.offsetHeight
  el.style.transitionProperty = ""
}
