"use client"

import { File as FileIcon, TriangleAlert, X } from "lucide-react"

import { formatSize } from "@/lib/attachments"
import { cn } from "@/lib/utils"

export interface PendingAttachment {
  /** Local, and stable across the upload — the server's id arrives later. */
  key: string
  file: File
  kind: "image" | "file"
  /** Object URL for image thumbnails. Caller owns revoking it. */
  previewUrl?: string
  status: "uploading" | "ready" | "failed"
  /** The stored row, once there is one. What a turn is sent with. */
  attachmentId?: string
  /** Why it failed, in the server's own words where it gave any. */
  error?: string
}

/**
 * The strip of attachments above the editor.
 *
 * A file is uploaded the moment it is picked, so a tile here is already a row
 * on the server unless it says otherwise — see chat-composer.tsx. What is
 * staged is only the *link* to the turn, which is made when the turn is sent.
 */
export function AttachmentStrip({
  attachments,
  onRemove,
}: {
  attachments: PendingAttachment[]
  onRemove: (key: string) => void
}) {
  if (attachments.length === 0) return null

  return (
    <div className="flex flex-wrap gap-2 px-3.5 pt-3">
      {attachments.map((a) => (
        <div
          key={a.key}
          className={cn(
            "group/attachment relative overflow-hidden rounded-(--radius-sheet) border border-border bg-muted",
            a.status === "failed" && "border-destructive/60"
          )}
          title={a.error ?? a.file.name}
        >
          {a.kind === "image" && a.previewUrl ? (
            // A transient local object URL — not worth Next's remote-image pipeline.
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={a.previewUrl}
              alt={a.file.name}
              className={cn(
                "size-14 object-cover transition-opacity",
                a.status !== "ready" && "opacity-50"
              )}
            />
          ) : (
            <div className="flex h-14 max-w-40 items-center gap-2 px-2.5">
              {a.status === "failed" ? (
                <TriangleAlert className="size-4 shrink-0 text-destructive" />
              ) : (
                <FileIcon className="size-4 shrink-0 text-muted-foreground" />
              )}
              <div className="min-w-0">
                <div className="truncate text-xs font-medium">
                  {a.file.name}
                </div>
                <div className="text-2xs text-muted-foreground">
                  {a.status === "failed"
                    ? "Not added"
                    : formatSize(a.file.size)}
                </div>
              </div>
            </div>
          )}

          {/* A hairline along the bottom edge rather than a mark on top of the
              tile: at 56px a centred spinner covers the thumbnail it is
              reporting on, and what the tile is waiting for is short enough
              that a rule reads better than a figure. */}
          {a.status === "uploading" && (
            <div className="pointer-events-none absolute inset-x-0 bottom-0 h-0.5 overflow-hidden bg-primary/20">
              <div className="h-full w-1/3 animate-[attachment-progress_1.1s_ease-in-out_infinite] bg-primary" />
            </div>
          )}

          <button
            type="button"
            onClick={() => onRemove(a.key)}
            aria-label={`Remove ${a.file.name}`}
            className={cn(
              "absolute top-1 right-1 flex size-4 items-center justify-center rounded-full bg-foreground/70 text-background opacity-0 transition-opacity",
              "group-hover/attachment:opacity-100 focus-visible:opacity-100",
              // A failure has nothing to hover for and one thing to do about it.
              a.status === "failed" && "bg-destructive opacity-100"
            )}
          >
            <X className="size-2.5" />
          </button>
        </div>
      ))}
    </div>
  )
}
