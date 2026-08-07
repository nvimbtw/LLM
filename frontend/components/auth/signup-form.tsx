"use client"

import Link from "next/link"

import { Button } from "@/components/ui/button"
import { FigureSeed } from "@/components/ui/figure-seed"
import { useSignup } from "@/hooks/use-signup"
import { AuthShell } from "./auth-shell"
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
    <AuthShell
      eyebrow="Create account"
      title="Start a workspace"
      description="One account gets you the chat, the model catalogue and your own conversation history."
      figure={<FigureSeed />}
      figureCaption="Begin with one model and a blank corpus. Everything after that is a run you queued."
      footer={
        <>
          Already have an account?{" "}
          <Link
            href="/login"
            className="font-medium text-foreground underline-offset-4 hover:underline"
          >
            Log in
          </Link>
        </>
      }
    >
      <form onSubmit={onSubmit} className="flex flex-col gap-5">
        {/* Labelled "Username", not "Name", because `use-signup.ts` posts this
            value as `username` and it is what the login screen will ask for. It
            used to say Name with an "Ada Lovelace" placeholder, which taught
            people to create an account they then could not log into. */}
        <AuthField
          id="username"
          label="Username"
          type="text"
          autoComplete="username"
          placeholder="adalovelace"
          hint="This is what you'll log in with."
          required
          autoFocus
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
        {error && (
          <p className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive">
            {error}
          </p>
        )}
        <Button type="submit" disabled={isSubmitting} className="h-10">
          {isSubmitting ? "Creating account…" : "Create account"}
        </Button>
      </form>
    </AuthShell>
  )
}
