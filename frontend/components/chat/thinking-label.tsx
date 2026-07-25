"use client"

import * as React from "react"

// Gem- and metalworking verbs, loosely Mohs-hardness themed — the models are
// named after minerals and metals, so "thinking" reads as the model working its
// stone. The actual reasoning is hidden; we just cycle these while it thinks, so
// the model can't get the indicator wrong.
const LABELS = [
  "Thinking",
  "Mining",
  "Quarrying",
  "Cleaving",
  "Carving",
  "Faceting",
  "Cutting",
  "Grinding",
  "Polishing",
  "Forging",
  "Tempering",
  "Annealing",
  "Smelting",
  "Refining",
  "Crystallizing",
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
          (prev + 1 + Math.floor(Math.random() * (LABELS.length - 1))) % LABELS.length
      )
    }, INTERVAL_MS)
    return () => clearInterval(id)
  }, [])

  return (
    // Fixed-width column. The labels range from "Mining" to "Crystallizing", and
    // the leader rule beside this fills whatever space is left — so without a
    // stable width the rule would visibly resize every time the word cycles.
    // Wide enough for the longest label at this size, with a little slack.
    <span className="text-shimmer inline-block min-w-[7.5rem] shrink-0 truncate font-medium">
      {LABELS[index]}…
    </span>
  )
}
