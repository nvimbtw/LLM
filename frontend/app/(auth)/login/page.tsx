import type { Metadata } from "next"
import Link from "next/link"

import { AuthShell } from "@/components/auth/auth-shell"
import { LoginForm } from "@/components/auth/login-form"

export const metadata: Metadata = { title: "Log in" }

export default function LoginPage() {
  return (
    <AuthShell
      title="Log in"
      footer={
        <>
          New to Slate?{" "}
          <Link
            href="/signup"
            className="text-foreground underline underline-offset-4"
          >
            Create account
          </Link>
        </>
      }
    >
      <LoginForm />
    </AuthShell>
  )
}
