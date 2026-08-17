import * as React from "react"
import Link from "next/link"

/**
 * The frame both auth screens are built in: the form on the paper, a figure on a
 * card panel beside it.
 *
 * Replaces `auth-card.tsx`, which was a shadcn `Card` — one of the last places
 * in the app still using it, and the reason login was the only screen not
 * speaking the editorial vocabulary. The type here is deliberately the same as
 * `page-header.tsx` and `panel.tsx` down to the values: `font-display` at
 * `text-3xl` for the title, the `tracking-[0.12em]` uppercase eyebrow, a hairline
 * `border-border` rule under the header. If those change there, they should
 * change here.
 *
 * The figure column is `hidden lg:flex`, not merely narrowed. Below `lg` a form
 * wants the whole width and an illustration is in the way; the screen resolves to
 * the centred column it always should have been rather than to a squeezed
 * version of the split.
 */
export interface AuthShellProps {
  /** Small uppercase label above the title — which of the two screens this is. */
  eyebrow: string
  title: string
  description: string
  children: React.ReactNode
  footer?: React.ReactNode
  /** The static SVG on the right. Hidden below `lg`. */
  figure: React.ReactNode
  /** One line under the figure. Set in the display face, so keep it short. */
  figureCaption: React.ReactNode
}

export function AuthShell({
  eyebrow,
  title,
  description,
  children,
  footer,
  figure,
  figureCaption,
}: AuthShellProps) {
  return (
    <div className="grid min-h-svh lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
      <div className="flex flex-col justify-center px-6 py-14 sm:px-12 lg:px-16">
        <div className="mx-auto w-full max-w-sm">
          {/* The wordmark doubles as the way out of the auth screens, since
              there is no sidebar here to get back to the app with. */}
          <Link
            href="/"
            className="inline-flex flex-col transition-opacity hover:opacity-70"
          >
            <span className="font-display text-xl leading-none font-semibold tracking-tight">
              Slate
            </span>
            <span className="mt-1 text-xs font-medium tracking-[0.22em] text-muted-foreground uppercase">
              LLM Project
            </span>
          </Link>

          <header className="mt-12 border-b border-border pb-6">
            <p className="text-xs font-medium tracking-[0.12em] text-muted-foreground uppercase">
              {eyebrow}
            </p>
            <h1 className="mt-2 font-display text-3xl font-semibold tracking-tight">
              {title}
            </h1>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              {description}
            </p>
          </header>

          <div className="mt-8">{children}</div>

          {footer && (
            <p className="mt-8 border-t border-border pt-6 text-sm text-muted-foreground">
              {footer}
            </p>
          )}
        </div>
      </div>

      <aside className="relative hidden flex-col justify-center border-l border-border bg-card px-16 lg:flex">
        <div className="mx-auto w-full max-w-md">
          {figure}
          <p className="mt-12 font-display text-lg leading-snug text-balance text-muted-foreground">
            {figureCaption}
          </p>
        </div>
      </aside>
    </div>
  )
}
