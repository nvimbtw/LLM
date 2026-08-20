"use client"

import * as React from "react"
import { FileText, ImageOff } from "lucide-react"

import { useAttachmentUrl } from "@/hooks/use-attachment-url"
import {
  extensionOf,
  formatSize,
  isImage,
  stemOf,
  type Attachment,
} from "@/lib/attachments"
import { cn } from "@/lib/utils"

/**
 * How tall the strip is. One height for both kinds of tile is what makes a
 * mixed row read as a row: a 112px thumbnail beside a 56px card is two things
 * that happen to be adjacent, and the ragged top edge is the first thing the
 * eye finds.
 */
const TILE = "h-28"

/**
 * The files sent with a question, above the bubble that asked it.
 *
 * Above rather than inside: the bubble is the words the person wrote, and a
 * thumbnail sitting in the same fill reads as part of the sentence. It ends on
 * the same right edge, so the whole turn is one column — the strip used to
 * stretch the full width and start at the far left, which put the attachments
 * and the question they belong to on opposite sides of the thread.
 */
export function MessageAttachments({
  attachments,
  onOpen,
  className,
}: {
  attachments: Attachment[]
  /** Absent leaves the tiles inert — a turn still streaming has nothing to show. */
  onOpen?: (attachment: Attachment) => void
  className?: string
}) {
  if (attachments.length === 0) return null

  return (
    <div
      className={cn(
        "mb-2 flex max-w-[min(52ch,85%)] flex-wrap justify-end gap-2",
        className
      )}
    >
      {attachments.map((attachment) => (
        <AttachmentTile
          key={attachment.id}
          attachment={attachment}
          onOpen={onOpen}
        />
      ))}
    </div>
  )
}

/**
 * One tile. A button rather than a div with a handler: this is the only way
 * into the viewer, so it has to be reachable from the keyboard and announce
 * itself as something that opens.
 */
function AttachmentTile({
  attachment,
  onOpen,
}: {
  attachment: Attachment
  onOpen?: (attachment: Attachment) => void
}) {
  return (
    <button
      type="button"
      disabled={!onOpen}
      onClick={() => onOpen?.(attachment)}
      aria-label={`Open ${attachment.filename}`}
      title={attachment.filename}
      className={cn(
        TILE,
        "group/tile relative overflow-hidden rounded-(--radius-sheet) border border-border/45 bg-card text-left shadow-sheet transition-all",
        // The tile lift the design system gives anything that is also a link.
        "enabled:hover:-translate-y-px enabled:hover:shadow-sheet-hover motion-reduce:transition-none motion-reduce:enabled:hover:translate-y-0",
        "focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
        "disabled:cursor-default"
      )}
    >
      {isImage(attachment) ? (
        <ImageTile attachment={attachment} />
      ) : (
        <FileTile attachment={attachment} />
      )}
    </button>
  )
}

/** A square of the picture itself. Square so a row of mixed shots stays a grid. */
function ImageTile({ attachment }: { attachment: Attachment }) {
  const { url, failed } = useAttachmentUrl(attachment.id)

  if (failed) {
    return (
      <div className="flex size-28 flex-col items-center justify-center gap-1.5 bg-muted text-muted-foreground">
        <ImageOff className="size-4" />
        <span className="text-2xs">Unavailable</span>
      </div>
    )
  }

  return (
    // The box is painted before the picture arrives and is the same shape after,
    // so a thread of mixed aspect ratios does not reflow as each one lands.
    <div className="size-28 bg-muted">
      {url && (
        // A local object URL for a private file — Next's image pipeline would
        // need a public URL to fetch, and there is none.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={url}
          alt={attachment.filename}
          className="size-full object-cover transition-transform duration-200 group-hover/tile:scale-[1.03] motion-reduce:transition-none motion-reduce:group-hover/tile:scale-100"
        />
      )}
    </div>
  )
}

/**
 * Anything that isn't a picture: a mark, the name, and what it is.
 *
 * The extension is lifted out of the name and into the meta line rather than
 * left on the end of it, because the end of the name is exactly what a card
 * this wide throws away — and "TXT · 5.0 MB" is the half a reader actually
 * needs when the title has been cut short.
 */
function FileTile({ attachment }: { attachment: Attachment }) {
  const extension = extensionOf(attachment.filename)

  return (
    <div className="flex h-full w-44 flex-col justify-between p-3">
      <FileText className="size-5 shrink-0 text-muted-foreground" />
      <div className="min-w-0">
        {/* Two lines, then an ellipsis. One line wastes the height the strip
            already has; three leaves no room for the meta line under it. */}
        <div className="line-clamp-2 text-xs leading-snug font-medium break-all">
          {stemOf(attachment.filename)}
        </div>
        <div className="mt-1 text-2xs text-muted-foreground">
          {extension ? `${extension.toUpperCase()} · ` : ""}
          {formatSize(attachment.size)}
        </div>
      </div>
    </div>
  )
}
