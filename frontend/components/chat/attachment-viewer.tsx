"use client"

import * as React from "react"
import { Download, ExternalLink, FileQuestion, X } from "lucide-react"

import { useAttachmentText } from "@/hooks/use-attachment-text"
import { useAttachmentUrl } from "@/hooks/use-attachment-url"
import {
  extensionOf,
  formatSize,
  previewKind,
  type Attachment,
} from "@/lib/attachments"
import { Button } from "@/components/ui/button"
import { Throbber } from "@/components/ui/throbber"
import { cn } from "@/lib/utils"

/**
 * An attachment, opened beside the conversation rather than over it.
 *
 * A panel and not a dialog, which is the whole point: what a person wants when
 * they open a file mid-thread is to read it *against* what was said about it.
 * A modal would take the conversation away to show its own attachment.
 *
 * Each kind of preview is its own component, because each needs a different
 * fetch — the text one wants the decoded contents, the other two want an object
 * URL — and a hook cannot be called conditionally. Mounting one of three is how
 * that stays honest.
 */
export function AttachmentViewer({
  attachment,
  onClose,
}: {
  attachment: Attachment
  onClose: () => void
}) {
  const kind = previewKind(attachment)

  // Escape closes it. The panel is not a dialog and traps nothing, so this is
  // the only key it owns — and closing on Escape is what every reader will try.
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose()
    }
    document.addEventListener("keydown", onKey)
    return () => document.removeEventListener("keydown", onKey)
  }, [onClose])

  return (
    <aside
      aria-label={`Preview of ${attachment.filename}`}
      className="flex h-full w-full min-w-0 shrink-0 flex-col border-l border-border bg-card lg:w-[26rem] xl:w-[32rem]"
    >
      <Header attachment={attachment} onClose={onClose} />

      {/* min-h-0 so the body scrolls inside the panel rather than pushing the
          header off the top of a column that has nowhere to grow. */}
      <div className="min-h-0 flex-1 overflow-auto">
        {kind === "image" && <ImagePreview attachment={attachment} />}
        {kind === "text" && <TextPreview attachment={attachment} />}
        {kind === "embed" && <EmbedPreview attachment={attachment} />}
        {kind === "none" && <NoPreview attachment={attachment} />}
      </div>
    </aside>
  )
}

/**
 * The name, what it is, and the two things you can do with it.
 *
 * Both actions need the bytes in hand, so they live here and the preview below
 * fetches its own — the same file, fetched twice, which the browser's cache
 * makes cheap and which keeps each half of the panel independently correct.
 */
function Header({
  attachment,
  onClose,
}: {
  attachment: Attachment
  onClose: () => void
}) {
  const { url } = useAttachmentUrl(attachment.id)
  const extension = extensionOf(attachment.filename)

  return (
    <div className="flex items-start gap-2 border-b border-border px-4 py-3">
      <div className="min-w-0 flex-1">
        <div
          className="truncate text-sm font-medium"
          title={attachment.filename}
        >
          {attachment.filename}
        </div>
        <div className="mt-0.5 text-2xs text-muted-foreground">
          {extension ? `${extension.toUpperCase()} · ` : ""}
          {formatSize(attachment.size)}
        </div>
      </div>

      {/* Absent until the bytes are here: a download that does nothing yet is
          worse than one that has not appeared.

          A bare `<a>` with `nativeButton={false}` rather than `ButtonLink`,
          which wraps `next/link` — these point at an object URL, which is not a
          route, and `download` is not one of Link's props. The prop is the same
          fix that component exists to remember. */}
      {url && (
        <>
          <Button
            variant="ghost"
            size="icon-sm"
            nativeButton={false}
            aria-label="Open in a new tab"
            render={<a href={url} target="_blank" rel="noreferrer" />}
          >
            <ExternalLink />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            nativeButton={false}
            aria-label={`Download ${attachment.filename}`}
            render={<a href={url} download={attachment.filename} />}
          >
            <Download />
          </Button>
        </>
      )}
      <Button
        variant="ghost"
        size="icon-sm"
        aria-label="Close the preview"
        onClick={onClose}
      >
        <X />
      </Button>
    </div>
  )
}

function ImagePreview({ attachment }: { attachment: Attachment }) {
  const { url, failed } = useAttachmentUrl(attachment.id)

  if (failed) return <Unavailable />
  if (!url) return <Loading />

  return (
    // A checkered-free plain ground and the picture contained, not covered:
    // this is the view where the whole of it matters, which is what the thumbnail
    // in the thread could not give.
    <div className="flex h-full items-center justify-center bg-muted/40 p-4">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={url}
        alt={attachment.filename}
        className="max-h-full max-w-full rounded-(--radius-sheet) object-contain"
      />
    </div>
  )
}

function TextPreview({ attachment }: { attachment: Attachment }) {
  const { text, withheld, failed } = useAttachmentText(attachment.id)

  if (failed) return <Unavailable />
  if (text === null) return <Loading />

  return (
    <div className="flex h-full flex-col">
      {withheld > 0 && (
        // Said rather than silently cut: a reader who scrolls to the end of a
        // novel and finds it stops mid-sentence has been lied to.
        <p className="border-b border-border bg-muted/40 px-4 py-2 text-2xs text-muted-foreground">
          Showing the first part of this file —{" "}
          {withheld.toLocaleString("en-US")} more characters are stored but not
          displayed. Download it to read the whole thing.
        </p>
      )}
      {/* `whitespace-pre-wrap` and not `pre`: a panel this narrow beside a
          conversation cannot also scroll sideways for every long line. */}
      <pre className="min-h-0 flex-1 overflow-auto px-4 py-3 font-mono text-xs leading-relaxed break-words whitespace-pre-wrap">
        {text}
      </pre>
    </div>
  )
}

/** A PDF, handed to the viewer the browser already has. */
function EmbedPreview({ attachment }: { attachment: Attachment }) {
  const { url, failed } = useAttachmentUrl(attachment.id)

  if (failed) return <Unavailable />
  if (!url) return <Loading />

  return (
    <iframe
      src={url}
      title={attachment.filename}
      className="size-full border-0 bg-muted/40"
    />
  )
}

function NoPreview({ attachment }: { attachment: Attachment }) {
  return (
    <Notice>
      <FileQuestion className="size-5" />
      <p>
        There is no preview for{" "}
        {extensionOf(attachment.filename) || "this kind of"}{" "}
        {extensionOf(attachment.filename) ? "files" : "file"}. It is stored, and
        the download above is the whole of it.
      </p>
    </Notice>
  )
}

function Unavailable() {
  return (
    <Notice>
      <FileQuestion className="size-5" />
      <p>This file could not be loaded.</p>
    </Notice>
  )
}

function Loading() {
  return (
    <div className="flex h-full items-center justify-center">
      <Throbber mode="loading" />
    </div>
  )
}

function Notice({ children }: { children: React.ReactNode }) {
  return (
    <div
      className={cn(
        "flex h-full flex-col items-center justify-center gap-3 px-8 text-center",
        "text-sm text-muted-foreground"
      )}
    >
      {children}
    </div>
  )
}
