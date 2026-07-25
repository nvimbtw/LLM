"use client"

import { FileText, FileType, File as FileIcon, X } from "lucide-react"

import { Button } from "@/components/ui/button"
import { formatBytes } from "@/components/files/constants"
import type { QueuedFile } from "@/hooks/use-file-upload"

function iconFor(name: string) {
  const lower = name.toLowerCase()
  if (lower.endsWith(".pdf")) return FileType
  if (lower.endsWith(".txt")) return FileText
  return FileIcon
}

export function FileList({
  files,
  onRemove,
  disabled,
}: {
  files: QueuedFile[]
  onRemove: (id: string) => void
  disabled?: boolean
}) {
  if (files.length === 0) return null

  return (
    <ul className="flex flex-col gap-2">
      {files.map(({ id, file }) => {
        const Icon = iconFor(file.name)
        return (
          <li
            key={id}
            className="flex items-center gap-3 rounded-md border bg-card px-3 py-2"
          >
            <Icon className="size-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium">{file.name}</p>
              <p className="text-xs text-muted-foreground">
                {formatBytes(file.size)}
              </p>
            </div>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="size-7 shrink-0"
              disabled={disabled}
              aria-label={`Remove ${file.name}`}
              onClick={() => onRemove(id)}
            >
              <X className="size-4" />
            </Button>
          </li>
        )
      })}
    </ul>
  )
}
