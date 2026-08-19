"use client"

import Link from "next/link"

import { Button } from "@/components/ui/button"
import { FigureLattice } from "@/components/ui/figure-lattice"
import { useLogin } from "@/hooks/use-login"
import { FormError } from "@/components/ui/error-state"
import { AuthShell } from "./auth-shell"
import { AuthField } from "./auth-field"

export function LoginForm() {
  const {
    username,
    setUsername,
    password,
    setPassword,
    isSubmitting,
    error,
    onSubmit,
  } = useLogin()

  return (
    <AuthShell
      eyebrow="Log in"
      title="Welcome back"
      description="Your models, runs and conversations are where you left them."
      figure={<FigureLattice />}
      figureCaption="Every token here was predicted by a model trained on this machine, on a corpus you chose."
      footer={
        <>
          Don&apos;t have an account?{" "}
          <Link
            href="/signup"
            className="font-medium text-foreground underline-offset-4 hover:underline"
          >
            Sign up
          </Link>
        </>
      }
    >
      <form onSubmit={onSubmit} className="flex flex-col gap-5">
        <AuthField
          id="username"
          label="Username"
          type="text"
          autoComplete="username"
          placeholder="adalovelace"
          required
          autoFocus
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
        {error && <FormError>{error}</FormError>}
        <Button type="submit" disabled={isSubmitting} className="h-10">
          {isSubmitting ? "Logging in…" : "Log in"}
        </Button>
      </form>
    </AuthShell>
  )
}
