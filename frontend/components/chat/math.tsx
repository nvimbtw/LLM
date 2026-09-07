"use client"

import * as React from "react"
import katex from "katex"
import "katex/dist/katex.min.css"

import { cn } from "@/lib/utils"

/**
 * One equation, typeset.
 *
 * KaTeX is synchronous and runs the same on the server as in the browser, so
 * this needs no effect and no post-mount swap: the markup is in the HTML that
 * first paints, and an equation never reflows the paragraph around it after the
 * fact — which matters most on the surface where the text is still arriving.
 *
 * `htmlAndMathml` (KaTeX's default) emits both the visual layout and a MathML
 * copy, the latter for screen readers; the visual half is aria-hidden by KaTeX
 * itself, so an equation is read as maths rather than spelled out as glyphs.
 */
export function Math({
  tex,
  display,
  className,
}: {
  tex: string
  display: boolean
  className?: string
}) {
  const html = React.useMemo(() => typeset(tex, display), [tex, display])

  if (html === null) {
    // TeX the parser accepted and KaTeX could not. Showing the source — in the
    // mono face, delimiters and all — says where the fault is; KaTeX's own
    // fallback prints the same string in alarm red, which reads as the app
    // failing rather than the equation being malformed.
    return (
      <span
        className={cn(
          "font-mono text-[0.85em] break-words text-muted-foreground",
          display && "mt-[0.9em] block first:mt-0"
        )}
        title="This equation could not be typeset"
      >
        {display ? `$$${tex}$$` : `$${tex}$`}
      </span>
    )
  }

  // A wide equation scrolls inside its own box rather than widening the reply;
  // `overflow-y` stays visible so a tall fraction is not clipped by the
  // scroller. Inline maths is left alone — it wraps with the line it sits in.
  return (
    <span
      className={cn(
        "rich-math",
        display &&
          "mt-[0.9em] block overflow-x-auto overflow-y-visible py-[0.15em] first:mt-0",
        className
      )}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}

// Typesetting the same equation on every streamed token is the one cost worth
// avoiding here: a reply re-renders per token, and the maths in it does not
// change once its closing delimiter has arrived. Bounded, because a long thread
// would otherwise keep every equation it has ever shown.
const CACHE_LIMIT = 512
const cache = new Map<string, string | null>()

/** The rendered HTML, or null when the TeX does not parse. */
function typeset(tex: string, display: boolean): string | null {
  const key = `${display ? "d" : "i"}:${tex}`
  const hit = cache.get(key)
  if (hit !== undefined) return hit

  let html: string | null
  try {
    html = katex.renderToString(tex, {
      displayMode: display,
      // Left to KaTeX to decide, rather than thrown: `strict` fires on things
      // like a Unicode minus or a stray `\newline`, which are worth rendering
      // anyway.
      strict: false,
      throwOnError: true,
    })
  } catch {
    html = null
  }

  if (cache.size >= CACHE_LIMIT) cache.clear()
  cache.set(key, html)
  return html
}
