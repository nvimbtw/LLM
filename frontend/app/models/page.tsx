import { ModelsGrid } from "@/components/models/models-grid";

export default function ModelsPage() {
  return (
    <div className="mx-auto w-full max-w-5xl flex-1 px-5 py-12 sm:px-8">
      <header className="mb-8 border-b border-border/60 pb-6">
        <h1 className="font-display text-3xl font-semibold tracking-tight">Models</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Every model available for your conversations, by family.
        </p>
      </header>
      <ModelsGrid />
    </div>
  );
}
