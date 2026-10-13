import { AuthScene } from "@/components/auth/auth-scene"

export default function AuthLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return <AuthScene>{children}</AuthScene>
}
