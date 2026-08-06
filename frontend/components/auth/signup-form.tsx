"use client"

import Link from "next/link"

import { Button } from "@/components/ui/button"
import { useSignup } from "@/hooks/use-signup"
import { AuthCard } from "./auth-card"
import { AuthField } from "./auth-field"

export function SignupForm() {
  const {
    name,
    setName,
    password,
    setPassword,
    isSubmitting,
    error,
    onSubmit,
  } = useSignup()

  return (
    <AuthCard
      title="Create your account"
      description="Sign up to start chatting."
      footer={
        <span>
          Already have an account?{" "}
          <Link
            href="/login"
            className="font-medium text-foreground underline-offset-4 hover:underline"
          >
            Log in
          </Link>
        </span>
      }
    >
      <form onSubmit={onSubmit} className="flex flex-col gap-4">
        <AuthField
          id="name"
          label="Name"
          type="text"
          autoComplete="name"
          placeholder="Ada Lovelace"
          required
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <AuthField
          id="password"
          label="Password"
          type="password"
          autoComplete="new-password"
          placeholder="••••••••"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {error && <p className="text-sm text-destructive">{error}</p>}
        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? "Creating account…" : "Sign up"}
        </Button>
      </form>
    </AuthCard>
  )
}
