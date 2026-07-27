import { cn } from "@/lib/utils"

/**
 * The two set-off blocks a reply can contain: a quotation and a warning.
 *
 * One component for both because they are the same object — a tinted panel with
 * a coloured left rail and an oversized mark bled into the right edge — differing
 * only in hue and glyph. Splitting them would mean two places to change when the
 * shape is adjusted, and they have already drifted once.
 *
 * The mark is decorative: it is `aria-hidden`, clipped by the container, and set
 * low enough in opacity to sit under the text rather than beside it. It exists so
 * the block is identifiable at a glance while scrolling, before any of it is
 * read — which is the entire job of a pull-quote marker.
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
        "relative my-4 overflow-hidden rounded-r-md border-l-2 py-3 pr-9 pl-4",
        t.rail,
        t.wash,
        t.text
      )}
    >
      {/* Small, and pinned to the top-right corner. It was previously 56px, which
          is display size — big enough to read as an illustration competing with
          the text rather than as a marker labelling it. At 16px it does the same
          job (tells you what kind of block this is at a glance) and stays out of
          the way of the sentence it is sitting beside. */}
      <span
        aria-hidden
        className={cn(
          "pointer-events-none absolute top-2.5 right-2.5 select-none",
          t.mark
        )}
      >
        {tone === "quote" ? <QuoteMark /> : <WarnMark />}
      </span>
      <div className="relative space-y-2">{children}</div>
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
