"use client"

import { useEffect, useState } from "react"
import { useAuth, useSessionToken } from "@/hooks/use-auth"
import { ServiceStatus, subscribeServiceHealth } from "@/lib/service-health"

/** Subscribe to service observations for the current administrator session only. */
export function useServiceHealth(): ServiceStatus[] | null {
  const token = useSessionToken()
  const { isAdmin } = useAuth()
  const [snapshot, setSnapshot] = useState<{
    token: string
    services: ServiceStatus[] | null
  } | null>(null)
  useEffect(() => {
    if (!token || !isAdmin) return
    return subscribeServiceHealth(token, (services) =>
      setSnapshot({ token, services })
    )
  }, [token, isAdmin])
  return token && isAdmin && snapshot?.token === token
    ? snapshot.services
    : null
}
