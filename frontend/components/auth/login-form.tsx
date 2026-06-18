"use client"

import Link from "next/link"

import { Button } from "@/components/ui/button"
import { useLogin } from "@/hooks/use-login"
import { AuthCard } from "./auth-card"
import { AuthField } from "./auth-field"

export function LoginForm() {
  const { username, setUsername, password, setPassword, isSubmitting, error, onSubmit } =
    useLogin()

  return (
    <AuthCard
      title="Welcome back"
      description="Log in to your account to continue."
      footer={
        <span>
          Don&apos;t have an account?{" "}
          <Link
            href="/signup"
            className="font-medium text-foreground underline-offset-4 hover:underline"
          >
            Sign up
          </Link>
        </span>
      }
    >
      <form onSubmit={onSubmit} className="flex flex-col gap-4">
        <AuthField
          id="username"
          label="Username"
          type="text"
          autoComplete="username"
          placeholder="adalovelace"
          required
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <AuthField
          id="password"
          label="Password"
          type="password"
          autoComplete="current-password"
          placeholder="••••••••"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {error && <p className="text-sm text-destructive">{error}</p>}
        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? "Logging in…" : "Log in"}
        </Button>
      </form>
    </AuthCard>
  )
}
