"use client"

import { Button } from "@/components/ui/button"
import { useLogin } from "@/hooks/use-login"
import { FormError } from "@/components/ui/error-state"
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
  )
}
