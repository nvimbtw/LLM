"use client"

import { Button } from "@/components/ui/button"
import { useSignup } from "@/hooks/use-signup"
import { FormError } from "@/components/ui/error-state"
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
      {error && <FormError>{error}</FormError>}
      <Button type="submit" disabled={isSubmitting} className="h-10">
        {isSubmitting ? "Creating account…" : "Create account"}
      </Button>
    </form>
  )
}
