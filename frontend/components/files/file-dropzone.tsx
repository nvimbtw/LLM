"use client"

import * as React from "react"
import { Upload } from "lucide-react"

import { cn } from "@/lib/utils"
import { ACCEPT_ATTR, ACCEPTED_EXTENSIONS } from "@/components/files/constants"

export function FileDropzone({
  onFiles,
  disabled,
}: {
  onFiles: (files: FileList | File[]) => void
  disabled?: boolean
}) {
  const inputRef = React.useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = React.useState(false)

  function handleDrop(event: React.DragEvent) {
    event.preventDefault()
    setDragging(false)
    if (disabled) return
    if (event.dataTransfer.files.length > 0) {
      onFiles(event.dataTransfer.files)
    }
  }

  function handleSelect(event: React.ChangeEvent<HTMLInputElement>) {
    if (event.target.files && event.target.files.length > 0) {
      onFiles(event.target.files)
    }
    // Reset so selecting the same file again still fires onChange.
    event.target.value = ""
  }

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => {
        if (!disabled && (e.key === "Enter" || e.key === " ")) {
          e.preventDefault()
          inputRef.current?.click()
        }
      }}
      onDragOver={(e) => {
        e.preventDefault()
        if (!disabled) setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={cn(
        "flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed px-6 py-10 text-center transition-colors",
        "outline-none focus-visible:ring-2 focus-visible:ring-ring",
        dragging ? "border-primary bg-primary/5" : "border-input bg-muted/30",
        disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer hover:bg-muted/50"
      )}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ACCEPT_ATTR}
        className="sr-only"
        disabled={disabled}
        onChange={handleSelect}
      />
      <div className="flex size-10 items-center justify-center rounded-full bg-muted text-muted-foreground">
        <Upload className="size-5" />
      </div>
      <p className="text-sm font-medium">
        Drag & drop files here, or click to browse
      </p>
      <p className="text-xs text-muted-foreground">
        {ACCEPTED_EXTENSIONS.join(", ")} — multiple files supported
      </p>
    </div>
  )
}
