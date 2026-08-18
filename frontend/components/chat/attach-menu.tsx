"use client"

import * as React from "react"
import { ImagePlus, Paperclip, Plus } from "lucide-react"

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Button } from "@/components/ui/button"

/**
 * The "+" at the bottom-left of the composer. One hidden file input serves
 * both menu items — its `accept` is set right before it's clicked, so
 * "Photos" and "Files" open the same picker filtered differently rather than
 * needing two inputs.
 */
export function AttachMenu({
  onFiles,
  disabled,
}: {
  onFiles: (files: FileList) => void
  disabled?: boolean
}) {
  const inputRef = React.useRef<HTMLInputElement>(null)

  const openPicker = (accept: string) => {
    const input = inputRef.current
    if (!input) return
    input.accept = accept
    input.click()
  }

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        multiple
        hidden
        onChange={(e) => {
          if (e.target.files?.length) onFiles(e.target.files)
          // Reset so picking the same file twice in a row still fires onChange.
          e.target.value = ""
        }}
      />
      <DropdownMenu>
        <DropdownMenuTrigger
          render={
            <Button
              type="button"
              variant="outline"
              size="icon-sm"
              className="rounded-full"
              aria-label="Add photos or files"
              disabled={disabled}
            >
              <Plus />
            </Button>
          }
        />
        <DropdownMenuContent side="top" align="start" className="min-w-44 p-1">
          <DropdownMenuItem onClick={() => openPicker("image/*")} className="gap-2">
            <ImagePlus className="text-muted-foreground" />
            Photos
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => openPicker("")} className="gap-2">
            <Paperclip className="text-muted-foreground" />
            Files
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </>
  )
}
