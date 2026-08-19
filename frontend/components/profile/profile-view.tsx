"use client"

import Link from "next/link"

import { Skeleton } from "@/components/ui/skeleton"
import { useChats } from "@/components/chat/chats-provider"
import { useAuth } from "@/hooks/use-auth"
import { useProfile } from "@/hooks/use-profile"
import { useUsage } from "@/hooks/use-usage"
import { LoadError } from "@/components/ui/error-state"
import { ChangePasswordForm } from "./change-password-form"
import { DeleteAccount } from "./delete-account"

/**
 * The account page.
 *
 * It used to be a username in a card and two destructive actions, which made the
 * whole page feel like a settings screen someone had forgotten to finish. The
 * facts below are all already fetched elsewhere in the session, so showing them
 * here costs nothing and gives the page a reason to be opened: who you are signed
 * in as, what that account has done, and only then how to change or end it.
 */
export function ProfileView() {
  const { user, isLoading, error, changePassword, deleteAccount } = useProfile()
  const { accessLevel } = useAuth()
  const { conversations } = useChats()
  const { usage } = useUsage()

  if (isLoading) {
    return (
      <div className="flex flex-col gap-6">
        <Skeleton className="h-28 w-full rounded-lg" />
        <Skeleton className="h-40 w-full rounded-lg" />
      </div>
    )
  }

  if (error || !user) {
    return <LoadError title={error ?? "Your profile didn't load."} />
  }

  return (
    <div className="flex flex-col gap-10">
      <section>
        <p className="text-xs font-medium tracking-[0.16em] text-muted-foreground uppercase">
          Signed in as
        </p>
        <h2 className="mt-1 font-display text-4xl font-semibold tracking-tight">
          {user.username}
        </h2>
        {accessLevel && (
          <p className="mt-2 text-sm text-muted-foreground">
            {accessLevel === "admin"
              ? "Admin — the queue, cycles and the corpus are yours to drive."
              : "Standard account."}
          </p>
        )}
      </section>

      {/* Counts rather than charts: /usage is one click away and does the charts
          properly. These are here to be glanced at. */}
      <section className="grid gap-px overflow-hidden rounded-lg border border-border bg-border/60 sm:grid-cols-3">
        <Stat
          label="Conversations"
          value={conversations.length.toLocaleString()}
          href="/conversations"
        />
        <Stat
          label="Requests · 30d"
          value={usage.totalRequests.toLocaleString()}
          href="/usage"
        />
        <Stat
          label="Tokens · 30d"
          value={compact(usage.totalTokens)}
          href="/usage"
        />
      </section>

      <ChangePasswordForm onSubmit={changePassword} />
      <DeleteAccount onDelete={deleteAccount} />
    </div>
  )
}

function Stat({
  label,
  value,
  href,
}: {
  label: string
  value: string
  href: string
}) {
  return (
    <Link
      href={href}
      className="bg-background px-5 py-4 transition-colors hover:bg-muted/40"
    >
      <p className="text-xs font-medium tracking-[0.16em] text-muted-foreground uppercase">
        {label}
      </p>
      <p className="mt-1 font-display text-2xl font-semibold tabular-nums">
        {value}
      </p>
    </Link>
  )
}

// Token counts run to seven figures and the exact digit is never the point here.
function compact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}
