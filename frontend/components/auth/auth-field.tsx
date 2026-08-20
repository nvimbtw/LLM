"use client"

import * as React from "react"
import { Eye, EyeOff } from "lucide-react"

import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"

/**
 * A labelled field on the auth screens.
 *
 * The label is the app's eyebrow — the `eyebrow` utility, muted — the same one
 * `panel.tsx` puts over a section, so a form field and a panel header are
 * recognisably the same idiom. It stays a plain `<label>` because there is
 * no `components/ui/label.tsx` in this codebase to reach for.
 *
 * A `type="password"` field grows a reveal button. The toggle swaps the input's
 * `type`, which is the only thing that actually works — a CSS-only version does
 * not exist, and remounting the input on toggle would drop the caret.
 */
export interface AuthFieldProps extends React.ComponentProps<typeof Input> {
  id: string
  label: string
  /** Optional line under the field — a constraint, not an error. */
  hint?: string
}

export function AuthField({
  id,
  label,
  hint,
  type,
  className,
  ...props
}: AuthFieldProps) {
  const [revealed, setRevealed] = React.useState(false)
  const isPassword = type === "password"

  return (
    <div className="flex flex-col gap-2">
      <label
        htmlFor={id}
        className="eyebrow text-muted-foreground"
      >
        {label}
      </label>
      <div className="relative">
        <Input
          id={id}
          type={isPassword && revealed ? "text" : type}
          className={cn("h-10", isPassword && "pr-10", className)}
          {...props}
        />
        {isPassword && (
          <button
            type="button"
            // Not in the tab order: it sits between the password field and the
            // submit button, and a keyboard user filling in a form wants the
            // next Tab to be "log in", not "show the password I just typed".
            tabIndex={-1}
            aria-label={revealed ? "Hide password" : "Show password"}
            onClick={() => setRevealed((v) => !v)}
            className="absolute inset-y-0 right-0 inline-flex w-10 items-center justify-center rounded-r-(--radius-control) text-muted-foreground transition-colors hover:text-foreground"
          >
            {revealed ? (
              <EyeOff className="size-4" />
            ) : (
              <Eye className="size-4" />
            )}
          </button>
        )}
      </div>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}
