import { LogoutButton } from "@/components/profile/logout-button";
import { ProfileView } from "@/components/profile/profile-view";

export default function ProfilePage() {
  return (
    <div className="mx-auto w-full max-w-2xl flex-1 px-4 py-8 sm:px-6">
      <header className="mb-8 flex items-start justify-between gap-4">
        <div>
          <h1 className="font-heading text-2xl font-semibold tracking-tight">
            Profile
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Manage your account settings.
          </p>
        </div>
        <LogoutButton />
      </header>
      <ProfileView />
    </div>
  );
}
