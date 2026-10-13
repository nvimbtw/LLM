"use client"

/* eslint-disable @next/next/no-img-element -- Draft previews use transient local object URLs. */

import { File as FileIcon, TriangleAlert, X } from "lucide-react"

import { formatSize } from "@/lib/attachments"
import { cn } from "@/lib/utils"

import type { PendingAttachment } from "@/lib/composer-attachments"
export type { PendingAttachment } from "@/lib/composer-attachments"

/**
 * The strip of attachments above the editor.
 *
 * A file is uploaded the moment it is picked, so a tile here is already a row
 * on the server unless it says otherwise — see chat-composer.tsx. What is
 * staged is only the *link* to the turn, which is made when the turn is sent.
 */
export interface AttachmentStripProps {
  attachments: PendingAttachment[]
  onRemove: (key: string) => void
}

/** Renders staged upload tiles and remove controls from the supplied draft; the caller owns their preview URLs. */
export function AttachmentStrip({
  attachments,
  onRemove,
}: AttachmentStripProps) {
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
