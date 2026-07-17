"use client"

import * as React from "react"
import { useRouter } from "next/navigation"

import { apiFetch, clearSession } from "@/lib/api"

export interface Profile {
  id: string
  username: string
}

export function useProfile() {
  const router = useRouter()

  const [user, setUser] = React.useState<Profile | null>(null)
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    let active = true

    async function load() {
      try {
        const response = await apiFetch(`/user/me`)
        if (!response.ok) {
          throw new Error("Unable to load your profile.")
        }
        const data: Profile = await response.json()
        if (active) setUser(data)
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : "Unable to load your profile.")
        }
      } finally {
        if (active) setIsLoading(false)
      }
    }

    load()
    return () => {
      active = false
    }
  }, [])

  async function changePassword(password: string) {
    if (!user) throw new Error("Profile not loaded yet.")
    const response = await apiFetch(`/user/${user.id}`, {
      method: "PUT",
      body: JSON.stringify({ username: user.username, password }),
    })
    if (!response.ok) {
      const message = await response.text()
      throw new Error(message || "Unable to update password.")
    }
  }

  async function deleteAccount() {
    if (!user) throw new Error("Profile not loaded yet.")
    const response = await apiFetch(`/user/${user.id}`, { method: "DELETE" })
    if (!response.ok) {
      const message = await response.text()
      throw new Error(message || "Unable to delete your account.")
    }
    clearSession()
    router.push("/login")
  }

  return { user, isLoading, error, changePassword, deleteAccount }
}
