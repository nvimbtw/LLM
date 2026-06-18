"use client"

import * as React from "react"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { AuthField } from "@/components/auth/auth-field"

export function ChangePasswordForm({
  onSubmit,
}: {
  onSubmit: (password: string) => Promise<void>
}) {
  const [password, setPassword] = React.useState("")
  const [confirm, setConfirm] = React.useState("")
  const [isSubmitting, setSubmitting] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [success, setSuccess] = React.useState(false)

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    setError(null)
    setSuccess(false)

    if (password !== confirm) {
      setError("Passwords do not match.")
      return
    }

    setSubmitting(true)
    try {
      await onSubmit(password)
      setPassword("")
      setConfirm("")
      setSuccess(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to update password.")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Change password</CardTitle>
        <CardDescription>Update the password for your account.</CardDescription>
      </CardHeader>
      <CardContent>
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
          {success && (
            <p className="text-sm text-muted-foreground">Password updated.</p>
          )}
          <Button type="submit" className="self-start" disabled={isSubmitting}>
            {isSubmitting ? "Saving…" : "Update password"}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
