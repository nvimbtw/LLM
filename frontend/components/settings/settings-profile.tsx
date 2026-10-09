"use client"

import { useAuth } from "@/hooks/use-auth"
import { useProfile } from "@/hooks/use-profile"
import { Skeleton } from "@/components/ui/skeleton"
import { LoadError } from "@/components/ui/error-state"
import { ChangePasswordForm } from "@/components/profile/change-password-form"
import { DeleteAccount } from "@/components/profile/delete-account"
import { LogoutButton } from "@/components/profile/logout-button"

export interface SettingsProfileProps {
  onClose: () => void
}

export function SettingsProfile({ onClose }: SettingsProfileProps) {
  const { user, isLoading, error, changePassword, deleteAccount } = useProfile()
  const { accessLevel } = useAuth()

  if (isLoading) return <Skeleton className="h-24 w-full" />
  if (error || !user)
    return <LoadError title={error ?? "Your profile didn't load."} />

  return (
    <section className="flex flex-col gap-6">
      <h2 className="font-display text-2xl">Profile</h2>
      <div className="flex flex-wrap items-center justify-between gap-3 border-y border-border py-4">
        <div className="min-w-0">
          <p className="text-sm font-medium break-words">{user.username}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            {accessLevel === "admin" ? "Administrator" : "Standard account"}
          </p>
        </div>
        <LogoutButton onLogout={onClose} />
      </div>
      <ChangePasswordForm compact onSubmit={changePassword} />
      <DeleteAccount
        compact
        onDelete={async () => {
          await deleteAccount()
          onClose()
        }}
      />
    </section>
  )
}
