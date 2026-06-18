"use client"

import * as React from "react"
import { useRouter } from "next/navigation"

import { useAuth } from "./use-auth"

export function useLogin() {
  const router = useRouter()
  const { login } = useAuth()

  const [username, setUsername] = React.useState("")
  const [password, setPassword] = React.useState("")
  const [isSubmitting, setSubmitting] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault()
    setSubmitting(true)
    setError(null)
    try {
      await login({ username, password })
      router.push("/")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to log in.")
    } finally {
      setSubmitting(false)
    }
  }

  return {
    username,
    setUsername,
    password,
    setPassword,
    isSubmitting,
    error,
    onSubmit,
  }
}
