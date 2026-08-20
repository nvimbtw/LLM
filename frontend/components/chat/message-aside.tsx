import { cn } from "@/lib/utils"

/**
 * The two set-off blocks a reply can contain: a quotation and a warning.
 *
 * One component for both because they are the same object — a tinted panel with
 * a coloured left rail and a small mark in the margin — differing only in hue and
 * glyph. Splitting them would mean two places to change when the shape is
 * adjusted, and they have already drifted once.
 *
 * The mark sits in the left gutter, aligned with the first line, rather than
 * floating in the top-right corner where it used to. In the corner it read as
 * decoration that had drifted there: nothing else in the app is pinned to a
 * corner, the block needed a lopsided `pr-9` to keep text out from under it, and
 * the eye met the tint, then the text, then a glyph somewhere off to the side.
 * In the margin it is what it actually is — a marginal mark labelling the block,
 * on the same axis as the rail, read before the first word rather than after the
 * first line.
 *
 * It is `aria-hidden` (the block's meaning is carried by its content, not by a
 * glyph), but it is not decorative in the design sense: hue alone is the weakest
 * way to tell two things apart and no way at all for some readers, so the mark is
 * what distinguishes a quote from a warning when the colours cannot.
 */
export type AsideTone = "quote" | "warn"

const TONE: Record<
  AsideTone,
  { rail: string; wash: string; mark: string; text: string }
> = {
  // Green, matching the existing quote treatment — kept so past replies don't
  // change colour under the reader.
  quote: {
    rail: "border-chart-2",
    wash: "bg-chart-2/8",
    // Higher opacity than the mark carried at display size: a 56px glyph at 20%
    // still reads, an 18px one at 20% is a smudge. Small marks need weight.
    mark: "text-chart-2/55",
    text: "text-muted-foreground",
  },
  // Amber rather than red: a warning is "read this carefully", not "this failed".
  // Destructive is spent on things that actually went wrong.
  warn: {
    rail: "border-kind-gather",
    wash: "bg-kind-gather/8",
    mark: "text-kind-gather/70",
    text: "text-foreground",
  },
}

export interface MessageAsideProps {
  tone: AsideTone
  children: React.ReactNode
}

export function MessageAside({ tone, children }: MessageAsideProps) {
  const t = TONE[tone]
  return (
    <div
      className={cn(
        // Margin in `em` so a set-off block keeps its distance from the prose
        // whatever size the surface set that prose at — the same rhythm every
        // other block in a reply is now on (see BLOCK in rich-message.tsx).
        // Symmetric padding now that nothing is pinned to the right edge; the
        // left inset is deeper because the mark lives in it.
        "relative my-[1.5em] overflow-hidden rounded-r-(--radius-control-sm) border-l-2 py-3 pr-4 pl-10",
        t.rail,
        t.wash,
        t.text
      )}
    >
      {/* In the gutter, and optically aligned with the first line rather than
          with the box: `top-3` matches the block's own `py-3`, so the mark sits
          on the first line's shoulder the way a marginal mark should. */}
      <span
        aria-hidden
        className={cn(
          "pointer-events-none absolute top-3 left-3 select-none",
          t.mark
        )}
      >
        {tone === "quote" ? <QuoteMark /> : <WarnMark />}
      </span>
      {/* No spacing of its own: the paragraphs inside carry their own leading,
          and a container gap on top of that spaced them twice. */}
      <div className="relative">{children}</div>
    </div>
  )
}

/**
 * A right double quotation mark, as two "66" glyphs — a bowl with a tail sweeping
 * out of it. Drawn rather than typed so it keeps its weight and proportions
 * whatever the display face is doing, and so it can be tinted independently of
 * the text.
 */
function QuoteMark() {
  // One glyph, drawn once and placed twice.
  const glyph =
    "M9.5 4C6.46 4 4 6.46 4 9.5S6.46 15 9.5 15c.3 0 .6-.03.9-.09-.94 2.42-3.03 4.24-5.6 4.72L5.6 22C10.42 21 14 16.74 14 11.65V9.5C14 6.46 12.54 4 9.5 4Z"
  return (
    // The viewBox is cropped to the two glyphs (18 wide each, 4–22 tall) so the
    // mark fills its 18px box instead of floating in a mostly-empty square.
    <svg width="18" height="18" viewBox="4 4 26 18" fill="currentColor">
      <path d={glyph} />
      <path d={glyph} transform="translate(12 0)" />
    </svg>
  )
}

/** An exclamation mark: a tapered bar over a dot. */
function WarnMark() {
  return (
    <svg width="14" height="18" viewBox="14 2 8 21" fill="currentColor">
      <path d="M16.2 3h3.6l-.7 13h-2.2l-.7-13Z" />
      <circle cx="18" cy="20.5" r="2.1" />
    </svg>
  )
}
