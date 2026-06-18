import { UsageOverview } from "@/components/usage/usage-overview";

export default function UsagePage() {
  return (
    <div className="mx-auto w-full max-w-5xl flex-1 px-4 py-8 sm:px-6">
      <header className="mb-8">
        <h1 className="font-heading text-2xl font-semibold tracking-tight">
          Usage
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Your requests, tokens, and cost over the last 30 days.
        </p>
      </header>
      <UsageOverview />
    </div>
  );
}
