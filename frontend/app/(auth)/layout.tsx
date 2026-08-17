import { ThemeToggle } from "@/components/theme-toggle"

/**
 * Bare frame for `/login` and `/signup`: no sidebar, no app header, no scroll
 * container — the auth screens are full-height compositions and own their own
 * layout. The theme toggle is here because it is the only chrome a logged-out
 * visitor has; inside the app it lives in the settings dialog.
 */
export default function AuthLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <div className="relative min-h-svh">
      <ThemeToggle className="absolute top-4 right-4 z-10" />
      {children}
    </div>
  )
}
