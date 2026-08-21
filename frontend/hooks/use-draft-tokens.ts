"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"

/** How long the composer sits still before asking what the draft costs. Short
 *  enough that the meter reads as live, long enough that a fast typist spends
 *  one request a phrase rather than one a letter. */
const SETTLE_MS = 200

/**
 * What the draft in the composer would cost, measured under the selected
 * model's own vocabulary rather than guessed from its length. The vocabularies
 * here are trained per model, so a characters-per-token guess is wrong by an
 * amount that changes with the model — against windows that start at 256
 * tokens, which is small enough for that to show.
 *
 * `text` is only the change signal. What is counted is whatever `read` returns
 * once typing settles, which is the markdown the turn will actually send and
 * not the plain text on screen.
 */
export function useDraftTokens(
  text: string,
  modelId: string | null,
  read: () => string
): number {
  const [tokens, setTokens] = React.useState(0)

  // Called when the pause happens rather than when the effect was set up, so
  // what gets counted is the document as it stands at that moment.
  const readRef = React.useRef(read)
  React.useEffect(() => {
    readRef.current = read
  })

  // An empty composer costs nothing, and saying so without a round trip is what
  // drops the meter back the instant a turn is sent. Adjusted during render
  // rather than in an Effect — the same pattern, for the same reason, as
  // `prevTitle` in composer-footer.tsx: it is derived from the props, so an
  // Effect would show the last count for a frame and then correct itself.
  const empty = !text.trim() || !modelId
  const [wasEmpty, setWasEmpty] = React.useState(empty)
  if (empty !== wasEmpty) {
    setWasEmpty(empty)
    if (empty) setTokens(0)
  }

  React.useEffect(() => {
    if (empty) return

    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      const draft = readRef.current()
      if (!draft.trim()) {
        setTokens(0)
        return
      }
      void countTokens(draft, modelId, controller.signal).then((counted) => {
        if (counted !== null) setTokens(counted)
      })
    }, SETTLE_MS)

    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [text, modelId, empty])

  return tokens
}

/**
 * `null` for anything that did not come back with a number — a request
 * superseded by the next keystroke, a model with no vocabulary built, the
 * engine down. The caller keeps the last count it had: a meter that fell to
 * zero mid-sentence would be a worse lie than one that is briefly stale.
 */
async function countTokens(
  text: string,
  modelId: string,
  signal: AbortSignal
): Promise<number | null> {
  try {
    const response = await apiFetch("/tokenize/count", {
      method: "POST",
      body: JSON.stringify({ model_name: modelId, text }),
      signal,
    })
    if (!response.ok) return null
    const data = (await response.json()) as { tokens?: number }
    return typeof data.tokens === "number" ? data.tokens : null
  } catch {
    return null
  }
}
