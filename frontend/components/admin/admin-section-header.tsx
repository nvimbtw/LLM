import type { ReactNode } from "react"

export interface AdminSectionHeaderProps {
  title: string
  meta?: ReactNode
  action?: ReactNode
}

export function AdminSectionHeader({
  title,
  meta,
  action,
}: AdminSectionHeaderProps) {
  return (
    <header className="mb-4 flex flex-wrap items-center justify-between gap-x-6 gap-y-4">
      <div className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 className="font-display text-2xl">{title}</h2>
        {meta && <p className="text-xs text-muted-foreground">{meta}</p>}
      </div>
      {action && (
        <div className="flex w-full items-center justify-end gap-4 sm:w-auto">
          {action}
        </div>
      )}
    </header>
  )
}
