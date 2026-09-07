"use client"

import * as React from "react"

import { usePrefersReducedMotion } from "@/hooks/use-reduced-motion"

// Plain verbs for what a model is plausibly doing while it reasons. Deliberately
// not a themed vocabulary: the app's name and its size classes don't imply one, and
// a metaphor here would have to be maintained forever and would age badly.
//
// The actual reasoning is hidden from us, so these are honest only in the weak sense
// that any of them could be true of a model mid-thought — which is why they stay
// general. "Thinking" is first because the first word typed must be deterministic,
// and it's also the safest of them.
const LABELS = [
  "Thinking",
  "Reading",
  "Weighing",
  "Considering",
  "Working",
  "Checking",
  "Comparing",
  "Narrowing",
  "Drafting",
  "Reviewing",
]

/** Dots after the word, cycling `.` → `..` → `...`. */
const DOTS = 3

/** How long a finished word stands, dots turning over, before the next one.
 * Long enough for two full turns of them at `DOT_MS`. */
const HOLD_MS = 3400

/**
 * The dots keep their own pace, and it is a slow one. They are not part of the
 * word's movement — the word is arriving, which is a thing with a beginning and
 * an end, while the dots are only saying that time is passing. Tying the two
 * together was tried, and running the dots at the speed a word types at made
 * the row twitch.
 */
const DOT_MS = 560

/**
 * A run — typing a word in, or turning one word into the next — starts at
 * `RUN_SLOW_MS` and accelerates to `RUN_FAST_MS` by its last step.
 */
const RUN_SLOW_MS = 155
const RUN_FAST_MS = 42

/**
 * The word beside the throbber while the model reasons — typed in a letter at a
 * time, then, every few seconds, turned into another word in place.
 *
 * Both movements are the same one operation, which is the whole trick here:
 * frame `i` shows `next.slice(0, i) + previous.slice(i)`, so the boundary
 * between the two words walks left to right and the length only ever changes at
 * that boundary. Typing in is that operation against an empty previous word.
 *
 *     Thinking… → Whinking. → Woinking.. → Wornking… → … → Working..
 *
 * The alternative — fade one word out, fade the next in — is a crossfade every
 * product has, and it also blanks the row for a moment, which beside a working
 * throbber reads as the work having stopped. This never blanks: something is
 * always in every character cell.
 *
 * The dots keep their own count across all of it, so they do not restart when a
 * word changes; they simply advance one step per frame like everything else.
 * They stay away during the very first typing-in, because a word that has not
 * finished arriving is not yet waiting on anything.
 */
export function ThinkingLabel() {
  const reduced = usePrefersReducedMotion()
  // Empty first render, on the server and on the client both: the first word is
  // typed rather than shown, and pre-rendering it whole would flash the finished
  // state for one frame before hydration takes it away again.
  const [frame, setFrame] = React.useState("")
  // The reduced-motion path's word, as an index rather than a rendered frame:
  // its first word is known during render, and a state written on mount to say
  // so would be a render's worth of work to arrive at what we already had.
  const [plain, setPlain] = React.useState(0)

  React.useEffect(() => {
    if (reduced) {
      // Reduced motion keeps the rotation — the row still has to say the model
      // is working — and drops only the per-character movement, which is the
      // part the preference is about.
      const id = setInterval(() => setPlain(nextWord), HOLD_MS)
      return () => clearInterval(id)
    }

    // Everything the loop needs, in the closure rather than in state: a frame is
    // scheduled off the last one, so a re-render must not be able to restart the
    // run half way through it.
    let timer = 0
    let word = 0
    let previous = ""
    let step = 1
    let steps = LABELS[0].length
    let dots = 0
    // Idle frames since the word settled, against HOLD_MS. Counted rather than
    // read off the dots, which turn over every three frames and so can never
    // measure anything longer than three of them.
    let held = 0
    // The dots hold off until the first word has finished arriving, and run
    // continuously from then on — including through every later word change.
    let settled = false

    const tick = () => {
      const target = LABELS[word]
      const running = step <= steps

      if (running) {
        if (settled) dots = (dots % DOTS) + 1
        setFrame(blend(target, previous, step) + ".".repeat(dots))
        step += 1
        timer = window.setTimeout(tick, runPace(step - 1, steps))
        return
      }

      // Settled: the dots turn over on their own until the hold is up, then the
      // next word starts walking through this one.
      settled = true
      dots = (dots % DOTS) + 1
      setFrame(target + ".".repeat(dots))

      held += 1
      if (held * DOT_MS < HOLD_MS) {
        timer = window.setTimeout(tick, DOT_MS)
        return
      }

      held = 0
      previous = target
      word = nextWord(word)
      step = 1
      steps = Math.max(LABELS[word].length, previous.length)
      timer = window.setTimeout(tick, runPace(1, steps))
    }

    timer = window.setTimeout(tick, runPace(1, steps))
    return () => clearTimeout(timer)
  }, [reduced])

  return (
    // Fixed-width column so the row does not breathe as the word changes length
    // mid-run. Sized for the longest label ("Considering") plus its dots at this
    // font size — and note the `truncate`, which means a width set too small
    // clips silently rather than complaining.
    <span className="text-shimmer inline-block min-w-[7rem] shrink-0 truncate font-medium">
      {reduced ? `${LABELS[plain]}…` : frame}
    </span>
  )
}

/** One frame of a word turning into another: the new word as far as `step`, the
 * old one from there on. Also the typewriter, when `previous` is empty. */
function blend(next: string, previous: string, step: number): string {
  return next.slice(0, step) + previous.slice(step)
}

/** A different word than the one showing, so a rotation never looks stuck. */
function nextWord(current: number): number {
  return (
    (current + 1 + Math.floor(Math.random() * (LABELS.length - 1))) %
    LABELS.length
  )
}

/** The gap before step `step` of `steps` — `RUN_SLOW_MS` at the first, and
 * `RUN_FAST_MS` by the last. */
function runPace(step: number, steps: number): number {
  return pace(steps > 1 ? (step - 1) / (steps - 1) : 1)
}

/**
 * The gap between frames at `t` through a movement: `RUN_SLOW_MS` at 0,
 * `RUN_FAST_MS` at 1, eased in and out so the movement gathers and settles
 * rather than starting and stopping at full tilt.
 *
 * Geometric between the two, not linear: these are speeds, and interpolating
 * the interval evenly spends most of the run near the fast end, where the
 * difference cannot be seen anyway.
 */
function pace(t: number): number {
  const u = Math.min(Math.max(t, 0), 1)
  const eased = u * u * (3 - 2 * u)
  return RUN_SLOW_MS * Math.pow(RUN_FAST_MS / RUN_SLOW_MS, eased)
}
