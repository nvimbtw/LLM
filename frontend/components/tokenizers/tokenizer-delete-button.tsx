"use client"

import * as React from "react"
import { Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { DeleteConfirmation } from "@/components/ui/delete-confirmation"
import type { TokenizerInfo } from "@/lib/tokenizers"

export interface TokenizerDeleteButtonProps {
  tokenizer: TokenizerInfo
  onDelete: () => void | Promise<void>
  iconOnly?: boolean
  disabled?: boolean
}

/** Confirms vocabulary deletion with explicit packed-corpus consequences, pending protection and retryable errors inside the dialog. */
export function TokenizerDeleteButton({
  tokenizer,
  onDelete,
  iconOnly = true,
  disabled = false,
}: TokenizerDeleteButtonProps) {
  const [open, setOpen] = React.useState(false)
  return (
    <>
      <Button
        type="button"
        variant="text"
        size={iconOnly ? "icon" : "sm"}
        aria-label={`Delete ${tokenizer.name}`}
        disabled={disabled}
        onClick={() => setOpen(true)}
        className="[@media(pointer:coarse)]:min-h-10 [@media(pointer:coarse)]:min-w-10"
      >
        <Trash2 className="size-4" />
        {!iconOnly && "Delete"}
      </Button>
      {open && (
        <DeleteConfirmation
          open
          onOpenChange={setOpen}
          title={`Delete ${tokenizer.name}?`}
          description="This removes the vocabulary and its packed corpora. Models trained with it keep their own vocabulary copy. This cannot be undone."
          label="Delete vocabulary"
          onDelete={async () => {
            await onDelete()
          }}
        />
      )}
    </>
  )
}
