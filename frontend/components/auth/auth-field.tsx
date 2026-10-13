"use client"

import * as React from "react"
import { Eye, EyeOff } from "lucide-react"

import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
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
  "aria-describedby": describedBy,
  ...props
}: AuthFieldProps) {
  const [revealed, setRevealed] = React.useState(false)
  const isPassword = type === "password"

  return (
    <div className="flex flex-col gap-2">
      <label htmlFor={id} className="eyebrow text-muted-foreground">
        {label}
      </label>
      <div className="relative">
        <Input
          variant="ruled"
          id={id}
          aria-describedby={
            [describedBy, hint && `${id}-hint`].filter(Boolean).join(" ") ||
            undefined
          }
          type={isPassword && revealed ? "text" : type}
          className={cn("h-10", isPassword && "pr-10", className)}
          {...props}
        />
        {isPassword && (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label={revealed ? "Hide password" : "Show password"}
            aria-pressed={revealed}
            aria-controls={id}
            onClick={() => setRevealed((v) => !v)}
            className="absolute top-1 right-0 text-muted-foreground hover:text-foreground"
          >
            {revealed ? (
              <EyeOff className="size-4" aria-hidden />
            ) : (
              <Eye className="size-4" aria-hidden />
            )}
          </Button>
        )}
      </div>
      {hint && (
        <p id={`${id}-hint`} className="text-xs text-muted-foreground">
          {hint}
        </p>
      )}
    </div>
  )
}
