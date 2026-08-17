import { LogoutButton } from "@/components/profile/logout-button"
import { ProfileView } from "@/components/profile/profile-view"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function ProfilePage() {
  return (
    <PageShell width="reading">
      <PageHeader
        title="Profile"
        description="Who you are signed in as, and what this account has been up to."
        action={<LogoutButton />}
      />
      <ProfileView />
    </PageShell>
  )
}
