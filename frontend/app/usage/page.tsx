import { UsageOverview } from "@/components/usage/usage-overview";

export default function UsagePage() {
  return (
    <div className="mx-auto w-full max-w-5xl flex-1 px-5 py-12 sm:px-8">
      <header className="mb-8 border-b border-border/60 pb-6">
        <h1 className="font-display text-3xl font-semibold tracking-tight">Usage</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Requests, tokens, and cost over the last 30 days.
        </p>
      </header>
      <UsageOverview />
    </div>
  );
}
