import { ModelsGrid } from "@/components/models/models-grid";

export default function ModelsPage() {
  return (
    <div className="mx-auto w-full max-w-5xl flex-1 px-4 py-8 sm:px-6">
      <header className="mb-8">
        <h1 className="font-heading text-2xl font-semibold tracking-tight">
          Models
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Browse the models available for your conversations.
        </p>
      </header>
      <ModelsGrid />
    </div>
  );
}
