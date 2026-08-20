"use client"

import { File as FileIcon, X } from "lucide-react"

import { cn } from "@/lib/utils"

export interface PendingAttachment {
  id: string
  file: File
  kind: "image" | "file"
  /** Object URL for image thumbnails. Caller owns revoking it. */
  previewUrl?: string
}

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/**
 * The strip of pending attachments above the editor. Purely a staging area
 * for now — see chat-composer.tsx, which discards these on send rather than
 * uploading them.
 */
export function AttachmentStrip({
  attachments,
  onRemove,
}: {
  attachments: PendingAttachment[]
  onRemove: (id: string) => void
}) {
  if (attachments.length === 0) return null

  return (
    <div className="flex flex-wrap gap-2 px-3.5 pt-3">
      {attachments.map((a) => (
        <div
          key={a.id}
          className="group/attachment relative overflow-hidden rounded-(--radius-sheet) border border-border bg-muted"
        >
          {a.kind === "image" && a.previewUrl ? (
            // A transient local object URL — not worth Next's remote-image pipeline.
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={a.previewUrl}
              alt={a.file.name}
              className="size-14 object-cover"
            />
          ) : (
            <div className="flex h-14 max-w-40 items-center gap-2 px-2.5">
              <FileIcon className="size-4 shrink-0 text-muted-foreground" />
              <div className="min-w-0">
                <div className="truncate text-xs font-medium">
                  {a.file.name}
                </div>
                <div className="text-2xs text-muted-foreground">
                  {formatSize(a.file.size)}
                </div>
              </div>
            </div>
          )}
          <button
            type="button"
            onClick={() => onRemove(a.id)}
            aria-label={`Remove ${a.file.name}`}
            className={cn(
              "absolute top-1 right-1 flex size-4 items-center justify-center rounded-full bg-foreground/70 text-background opacity-0 transition-opacity",
              "group-hover/attachment:opacity-100 focus-visible:opacity-100"
            )}
          >
            <X className="size-2.5" />
          </button>
        </div>
      ))}
    </div>
  )
}
