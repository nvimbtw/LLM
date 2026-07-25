import { ThrobberShowcase } from "@/components/throbber/throbber-showcase";

export default function ThrobberPage() {
  return (
    <div className="mx-auto w-full max-w-3xl flex-1 px-5 py-12 sm:px-8">
      <header className="mb-8 border-b border-border/60 pb-6">
        <h1 className="font-display text-3xl font-semibold tracking-tight">Throbber</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Ledger ticks — three graduations in a fixed footprint. Every state is
          the same three marks with different numbers, so they morph rather than
          swap.
        </p>
      </header>
      <ThrobberShowcase />
    </div>
  );
}
