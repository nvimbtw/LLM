import { BenchmarkView } from "@/components/benchmark/benchmark-view";

export default function BenchmarkPage() {
  return (
    <div className="mx-auto w-full max-w-5xl flex-1 px-5 py-12 sm:px-8">
      <header className="mb-8 border-b border-border/60 pb-6">
        <h1 className="font-display text-3xl font-semibold tracking-tight">Benchmark</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Held-out perplexity, minimal-pair grammar, and surface metrics — numbers
          instead of an impression of how close to English a model sounds.
        </p>
      </header>
      <BenchmarkView />
    </div>
  );
}
