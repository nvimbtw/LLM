"use client"

import type { ReactNode } from "react"
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "./dialog"

export const ACCOUNT_MODAL_PANEL =
  "min-w-0 overflow-y-auto px-4 py-8 data-ending-style:hidden sm:px-8"
export const ACCOUNT_MODAL_TAB =
  "h-auto flex-none px-3 py-2 text-xs font-normal after:bg-primary group-data-vertical/tabs:after:inset-y-2 group-data-vertical/tabs:after:right-auto group-data-vertical/tabs:after:left-0 group-data-vertical/tabs:after:w-px"

/** Shares the exact settings/notifications dialog dimensions, surface and accessible naming. */
export function AccountModal({
  open,
  onOpenChange,
  title,
  description,
  children,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  description: string
  children: ReactNode
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="h-[min(36rem,calc(100dvh-2rem))] max-w-3xl gap-0 overflow-hidden bg-background p-0">
        <DialogTitle className="sr-only">{title}</DialogTitle>
        <DialogDescription className="sr-only">{description}</DialogDescription>
        {children}
      </DialogContent>
    </Dialog>
  )
}

/** Keeps the account modal's section rail fixed while its neighboring content scrolls. */
export function AccountModalRail({
  title,
  shortcut,
  children,
}: {
  title: string
  shortcut: string
  children: ReactNode
}) {
  return (
    <aside className="flex w-28 shrink-0 flex-col border-r border-border px-2 py-8 sm:w-40 sm:px-4">
      <p className="mb-6 px-2 eyebrow text-muted-foreground">{title}</p>
      {children}
      <kbd className="mt-auto px-2 pt-6 font-mono text-xs text-muted-foreground/60">
        {shortcut}
      </kbd>
    </aside>
  )
}
