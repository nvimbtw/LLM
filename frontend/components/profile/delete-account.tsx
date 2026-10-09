"use client"

import * as React from "react"

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
import { Button } from "@/components/ui/button"
import { FormError } from "@/components/ui/error-state"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

export interface DeleteAccountProps {
  onDelete: () => Promise<void>
  compact?: boolean
}

export function DeleteAccount({
  onDelete,
  compact = false,
}: DeleteAccountProps) {
  const [isDeleting, setDeleting] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  async function handleDelete() {
    setDeleting(true)
    setError(null)
    try {
      await onDelete()
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unable to delete your account."
      )
      setDeleting(false)
    }
  }

  return (
    <Card
      className={
        compact
          ? "rounded-none border-t border-border bg-transparent pt-5 pb-0 shadow-none ring-0"
          : "border-destructive/40"
      }
    >
      <CardHeader className={compact ? "px-0" : undefined}>
        <CardTitle>Delete account</CardTitle>
        <CardDescription className={compact ? "text-xs" : undefined}>
          Permanently delete your account. This action cannot be undone.
        </CardDescription>
      </CardHeader>
      <CardContent
        className={compact ? "flex flex-col gap-3 px-0" : "flex flex-col gap-3"}
      >
        {error && <FormError>{error}</FormError>}
        <AlertDialog>
          <AlertDialogTrigger
            render={
              <Button
                variant={compact ? "ghost" : "destructive"}
                className={
                  compact
                    ? "self-start text-muted-foreground hover:text-destructive"
                    : "self-start"
                }
              >
                Delete account
              </Button>
            }
          />
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
              <AlertDialogDescription>
                This permanently deletes your account and signs you out. This
                action cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                onClick={handleDelete}
                disabled={isDeleting}
                className="bg-destructive text-white hover:bg-destructive/90"
              >
                {isDeleting ? "Deleting…" : "Delete account"}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </CardContent>
    </Card>
  )
}
