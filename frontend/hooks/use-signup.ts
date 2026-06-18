"use client"

import * as React from "react"
import { useRouter } from "next/navigation"

import { API_URL } from "@/lib/api"

export function useSignup() {
  const router = useRouter()

  const [name, setName] = React.useState("")
  const [password, setPassword] = React.useState("")
  const [isSubmitting, setSubmitting] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault()
    setSubmitting(true)
    setError(null)
    try {
      const response = await fetch(`${API_URL}/user`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: name, password }),
      })

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || "Unable to create account.")
      }

      router.push("/login")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to create account.")
    } finally {
      setSubmitting(false)
    }
  }

  return {
    name,
    setName,
    password,
    setPassword,
    isSubmitting,
    error,
    onSubmit,
  }
}
