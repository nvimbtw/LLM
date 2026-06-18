"use client"

import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { useProfile } from "@/hooks/use-profile"
import { ChangePasswordForm } from "./change-password-form"
import { DeleteAccount } from "./delete-account"

export function ProfileView() {
  const { user, isLoading, error, changePassword, deleteAccount } = useProfile()

  if (isLoading) {
    return <Skeleton className="h-32 w-full" />
  }

  if (error || !user) {
    return (
      <p className="text-sm text-destructive">
        {error ?? "Unable to load your profile."}
      </p>
    )
  }

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <CardHeader>
          <CardDescription>Username</CardDescription>
          <CardTitle className="text-xl">{user.username}</CardTitle>
        </CardHeader>
      </Card>

      <ChangePasswordForm onSubmit={changePassword} />
      <DeleteAccount onDelete={deleteAccount} />
    </div>
  )
}
