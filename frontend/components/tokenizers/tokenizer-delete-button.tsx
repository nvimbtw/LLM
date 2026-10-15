"use client"

import { Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { TokenizerInfo } from "@/lib/tokenizers"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"

export interface TokenizerDeleteButtonProps {
  tokenizer: TokenizerInfo
  onDelete: () => void | Promise<void>
  iconOnly?: boolean
  disabled?: boolean
}

/** Confirms vocabulary deletion with its packed-corpus consequences before invoking the caller's write. */
export function TokenizerDeleteButton({
  tokenizer,
  onDelete,
  iconOnly = true,
  disabled = false,
}: TokenizerDeleteButtonProps) {
  return (
    <AlertDialog>
      <AlertDialogTrigger
        render={
          <Button
            variant="ghost"
            size={iconOnly ? "icon-sm" : "sm"}
            aria-label={`Delete ${tokenizer.name}`}
            disabled={disabled}
          >
            <Trash2 className="size-4" />
            {!iconOnly && "Delete"}
          </Button>
        }
      />
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete {tokenizer.name}?</AlertDialogTitle>
          <AlertDialogDescription>
            The corpora packed under it go too — without this vocabulary they
            are token ids nothing can read. Models trained with it keep their
            own copy and are unaffected.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={() => void onDelete()}
            className="bg-destructive text-white hover:bg-destructive/90"
          >
            Delete
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
