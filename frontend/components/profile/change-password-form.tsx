"use client"

import * as React from "react"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { toast } from "@/components/ui/toast"
import { AuthField } from "@/components/auth/auth-field"

export interface ChangePasswordFormProps {
  onSubmit: (password: string) => Promise<void>
  compact?: boolean
}

export function ChangePasswordForm({
  onSubmit,
  compact = false,
}: ChangePasswordFormProps) {
  const [password, setPassword] = React.useState("")
  const [confirm, setConfirm] = React.useState("")
  const [isSubmitting, setSubmitting] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    setError(null)

    // A mismatch is about the two fields in front of you, so it stays inline.
    if (password !== confirm) {
      setError("Passwords do not match.")
      return
    }

    setSubmitting(true)
    try {
      await onSubmit(password)
      setPassword("")
      setConfirm("")
      toast.success("Password updated.")
    } catch (err) {
      toast.error(
        "Couldn't update your password",
        err instanceof Error ? err.message : "Something went wrong."
      )
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Card
      className={
        compact
          ? "rounded-none bg-transparent py-0 shadow-none ring-0"
          : undefined
      }
    >
      <CardHeader className={compact ? "px-0" : undefined}>
        <CardTitle>Change password</CardTitle>
      </CardHeader>
      <CardContent className={compact ? "px-0" : undefined}>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <AuthField
            id="new-password"
            label="New password"
            type="password"
            autoComplete="new-password"
            placeholder="••••••••"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <AuthField
            id="confirm-password"
            label="Confirm new password"
            type="password"
            autoComplete="new-password"
            placeholder="••••••••"
            required
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
          />
          {error && <p className="text-sm text-destructive">{error}</p>}
          <Button type="submit" className="self-start" disabled={isSubmitting}>
            {isSubmitting ? "Saving…" : "Update password"}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
