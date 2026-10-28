"use client"

import type { ReactNode } from "react"
import { useDeleteConfirmation } from "@/hooks/use-delete-confirmation"
import { FormError } from "./error-state"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "./alert-dialog"

export interface DeleteConfirmationProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  description: ReactNode
  label: string
  onDelete: () => Promise<void>
  onDeleted?: () => void
}

/** Confirms destructive consequences with shared theme-aware controls, pending protection and in-dialog retryable errors. */
export function DeleteConfirmation({
  open,
  onOpenChange,
  title,
  description,
  label,
  onDelete,
  onDeleted,
}: DeleteConfirmationProps) {
  const action = useDeleteConfirmation(onDelete, () => {
    onOpenChange(false)
    onDeleted?.()
  })
  return (
    <AlertDialog
      open={open}
      onOpenChange={(next) => {
        if (!action.pending) onOpenChange(next)
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          <AlertDialogDescription>{description}</AlertDialogDescription>
        </AlertDialogHeader>
        {action.error && <FormError>{action.error}</FormError>}
        <AlertDialogFooter>
          <AlertDialogCancel disabled={action.pending}>
            Cancel
          </AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            disabled={action.pending}
            onClick={() => void action.confirm()}
          >
            {action.pending ? "Deleting…" : label}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
