"use client"

import { Fragment, type ReactNode } from "react"
import { useAuth, useSessionToken } from "@/hooks/use-auth"
import { LoadError } from "@/components/ui/error-state"
import { Skeleton } from "@/components/ui/skeleton"

export interface AdminBoundaryProps {
  children: ReactNode
}

/** Mounts administrative workspaces/composers only for the actual administrator session; Rust remains the authorization boundary. */
export function AdminBoundary({ children }: AdminBoundaryProps) {
  const { isAdmin, isLoading } = useAuth()
  const token = useSessionToken()
  if (isLoading) return <Skeleton className="h-64 w-full" />
  if (!isAdmin || !token)
    return <LoadError title="Administrator access required." />
  return <Fragment key={token}>{children}</Fragment>
}
