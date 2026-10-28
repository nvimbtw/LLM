"use client"

import * as React from "react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { FormError } from "@/components/ui/error-state"
import { toast } from "@/components/ui/toast"
import { AuthField } from "@/components/auth/auth-field"

export interface ChangePasswordFormProps {
  onSubmit: (password: string) => Promise<void>
  compact?: boolean
}

/** Validates password confirmation, reports failures through notifications and preserves invalid-field focus. */
export function ChangePasswordForm({
  onSubmit,
  compact = false,
}: ChangePasswordFormProps) {
  const [password, setPassword] = React.useState("")
  const [confirm, setConfirm] = React.useState("")
  const [isSubmitting, setSubmitting] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  /** Rejects mismatched confirmation locally; submits matching values once and clears them only after success. */
  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    setError(null)

    if (password !== confirm) {
      setError("Passwords do not match.")
      event.currentTarget
        .querySelector<HTMLInputElement>("#confirm-password")
        ?.focus()
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
            onChange={(e) => {
              setError(null)
              setPassword(e.target.value)
            }}
          />
          <AuthField
            id="confirm-password"
            label="Confirm new password"
            type="password"
            autoComplete="new-password"
            placeholder="••••••••"
            required
            value={confirm}
            aria-invalid={Boolean(error)}
            aria-describedby={error ? "password-confirmation-error" : undefined}
            onChange={(e) => {
              setError(null)
              setConfirm(e.target.value)
            }}
          />
          {error && (
            <span id="password-confirmation-error">
              <FormError>{error}</FormError>
            </span>
          )}
          <Button type="submit" className="self-start" disabled={isSubmitting}>
            {isSubmitting ? "Saving…" : "Update password"}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
