"use client"

import * as React from "react"

// Plain verbs for what a model is plausibly doing while it reasons. Deliberately
// not a themed vocabulary: the app's name and its size classes don't imply one, and
// a metaphor here would have to be maintained forever and would age badly.
//
// The actual reasoning is hidden from us, so these are honest only in the weak sense
// that any of them could be true of a model mid-thought — which is why they stay
// general. "Thinking" is first because the first render must be deterministic (see
// below), and it's also the safest of them.
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

const INTERVAL_MS = 2200

export function ThinkingLabel() {
  // Start deterministically (no hydration mismatch); rotate on the client.
  const [index, setIndex] = React.useState(0)

  React.useEffect(() => {
    const id = setInterval(() => {
      // Jump by a random step so it never repeats a word back-to-back and
      // doesn't feel like a fixed loop.
      setIndex(
        (prev) =>
          (prev + 1 + Math.floor(Math.random() * (LABELS.length - 1))) %
          LABELS.length
      )
    }, INTERVAL_MS)
    return () => clearInterval(id)
  }, [])

  return (
    // Fixed-width column, so the token count to the right of this doesn't shift
    // every time the word cycles. Sized for the longest label ("Considering") at this
    // font size with a little slack — and note the `truncate`, which means a width
    // set too small clips silently rather than complaining.
    <span className="text-shimmer inline-block min-w-[6.25rem] shrink-0 truncate font-medium">
      {LABELS[index]}…
    </span>
  )
}
