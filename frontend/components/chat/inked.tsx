"use client"

import * as React from "react"

/**
 * The unit an animation is hung on: a word together with the whitespace in
 * front of it, which is the shape a token actually arrives in.
 *
 * The LLM engine yields byte-level BPE fragments — `llm-engine/llm/orchestrate/chat.py`
 * notes that one "can be half a UTF-8 character" — and a leading space rides
 * with the word it belongs to rather than arriving as anything of its own. This
 * used to split on `/(\s+)/`, which took that space back out and rendered it as
 * a bare text node between two animated spans: twice the nodes, and a space
 * that had to sit un-animated between them until its word turned up. Matching
 * `\s*\S+` keeps the pair together.
 *
 * A run of whitespace with no word after it yet still matches on its own and is
 * rendered bare — that is the one frame where a space has arrived and its word
 * has not. It is absorbed into the span on the next token.
 *
 * The regex is module-level and global, which is only safe because
 * `String.prototype.match` sets `lastIndex` to 0 before it starts. Do not reach
 * for `.exec` or `.test` with it.
 */
const TOKEN = /\s*\S+|\s+/g

/**
 * A settled prefix of a streaming text run.
 *
 * `word-ink` is an arrival animation, not a styling: once it has played the
 * span holds nothing the characters themselves do not. Rendering the settled
 * part as plain text is therefore identical on screen — and it is memoised, so
 * a reply that has grown to a hundred thousand characters still costs one text
 * node per frame instead of one span per word.
 */
const SettledText = React.memo(function SettledText({
  text,
}: {
  text: string
}) {
  return <>{text}</>
})

/**
 * How much trailing text keeps its arrival animation.
 *
 * Anything nearer the end may still be mid-animation, so it stays a live run of
 * spans; the window is roughly a long line and outlasts `word-ink-in`, which
 * means a word reaches the settled prefix only after it has finished moving.
 */
const LIVE_WINDOW = 240

/** Offset where the still-animating tail begins; 0 while the whole run is live. */
function liveStart(text: string): number {
  if (text.length <= LIVE_WINDOW) return 0
  const from = text.length - LIVE_WINDOW
  const at = Math.max(text.lastIndexOf(" ", from), text.lastIndexOf("\n", from))
  return at < 0 ? 0 : at + 1
}

/**
 * The live edge: text with each arrival unit wrapped so a newly-arrived one is
 * *set* — arriving a little loose and tightening to its own tracking, which is
 * what `word-ink` in `globals.css` draws.
 *
 * Keyed by index deliberately. React keeps the units already on screen mounted,
 * so only the ones this render added animate; keying by content would remount a
 * word every time the same one arrived again later in the sentence, and the
 * reply would twinkle. A unit that has something after it is final, which is
 * exactly the boundary `Inked` settles on.
 *
 * The word is still an approximation of the arrival unit and knowingly so: the
 * animation fires on a word's *first* fragment, and the rest of the word grows
 * inside a span that is already mid-movement. Waiting for the word to complete
 * is worse — it animates type the reader can already read — and animating the
 * true unit would mean threading token offsets through the parser, since
 * `RichContent` is handed the whole accumulated string.
 *
 * Note for whoever comes next: pairing the space with the word does *not*
 * narrow the leading edge. Measured against a settled line, the tracking pushes
 * it about 21px wide at 26px type either way, and carrying the space inside the
 * span tracks the space too, which is marginally wider still. The knob for how
 * far the edge travels is the 0.055em delta in `globals.css`, not this split.
 */
function LiveInk({ text }: { text: string }) {
  return (
    <>
      {(text.match(TOKEN) ?? []).map((part, i) =>
        part.trim() === "" ? (
          part
        ) : (
          <span key={i} className="word-ink">
            {part}
          </span>
        )
      )}
    </>
  )
}

/**
 * Text of one streaming run: the settled prefix, which is plain, and the live
 * edge, which is animated word by word.
 *
 * Splitting them is what keeps a growing paragraph from costing one span per
 * word per frame; see `SettledText` and `LIVE_WINDOW`.
 */
export function Inked({ text }: { text: string }) {
  const live = liveStart(text)
  if (live === 0) return <LiveInk text={text} />
  return (
    <>
      <SettledText text={text.slice(0, live)} />
      <LiveInk text={text.slice(live)} />
    </>
  )
}
