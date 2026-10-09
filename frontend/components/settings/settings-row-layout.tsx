import type * as React from "react"

export interface SettingsRowLayoutProps {
  label: string
  control?: React.ReactNode
  below?: React.ReactNode
  htmlFor?: string
}

export function SettingsRowLayout({
  label,
  control,
  below,
  htmlFor,
}: SettingsRowLayoutProps) {
  return (
    <div>
      <div className="flex flex-col items-start justify-between gap-3 sm:flex-row sm:gap-6">
        <div className="min-w-0">
          <label
            htmlFor={htmlFor}
            className="text-sm font-medium text-foreground"
          >
            {label}
          </label>
        </div>
        {control && <div className="max-w-full shrink-0 pt-0.5">{control}</div>}
      </div>
      {below && <div className="mt-2">{below}</div>}
    </div>
  )
}
