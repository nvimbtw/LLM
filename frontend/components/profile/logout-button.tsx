"use client"

import { LogOut } from "lucide-react"
import { useRouter } from "next/navigation"

import { Button } from "@/components/ui/button"
import { useAuth } from "@/hooks/use-auth"

export function LogoutButton() {
  const router = useRouter()
  const { logout } = useAuth()

  async function handleLogout() {
    await logout()
    router.push("/login")
  }

  return (
    <Button variant="outline" onClick={handleLogout}>
      <LogOut />
      Log out
    </Button>
  )
}
