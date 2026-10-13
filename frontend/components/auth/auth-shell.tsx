import type { ReactNode } from "react"
import Link from "next/link"

export interface AuthShellProps {
  title: string
  children: ReactNode
  footer: ReactNode
}

export function AuthShell({ title, children, footer }: AuthShellProps) {
  return (
    <section className="w-full max-w-sm" aria-labelledby="auth-title">
      <Link
        href="/"
        className="mb-10 inline-block font-display text-xl text-muted-foreground outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
      >
        Slate
      </Link>
      <h1 id="auth-title" className="mb-8 font-display text-3xl">
        {title}
      </h1>
      {children}
      <p className="mt-8 border-t border-border pt-6 text-sm text-muted-foreground">
        {footer}
      </p>
    </section>
  )
}
