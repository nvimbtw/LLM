"use client"

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
 * Text with each of those units wrapped so a newly-arrived one is *set* —
 * arriving a little loose and tightening to its own tracking, which is what
 * `word-ink` in `globals.css` draws.
 *
 * Keyed by index deliberately. React keeps the units already on screen mounted,
 * so only the ones this render added animate; keying by content would remount a
 * word every time the same one arrived again later in the sentence, and the
 * reply would twinkle. A settled unit is never re-created — its chunk can only
 * grow while it is the last one, and a chunk that has something after it is
 * final.
 *
 * The word is still an approximation of the arrival unit and knowingly so: the
 * animation fires on a word's *first* fragment, and the rest of the word grows
 * inside a span that is already mid-movement. Waiting for the word to complete
 * is worse — it animates type the reader can already read — and animating the
 * true unit would mean threading token offsets through the parser, since
 * `RichContent` is handed the whole accumulated string and re-parses it.
 *
 * Note for whoever comes next: pairing the space with the word does *not*
 * narrow the leading edge. Measured against a settled line, the tracking pushes
 * it about 21px wide at 26px type either way, and carrying the space inside the
 * span tracks the space too, which is marginally wider still. The knob for how
 * far the edge travels is the 0.055em delta in `globals.css`, not this split.
 */
export function Inked({ text }: { text: string }) {
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
