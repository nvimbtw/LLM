import { CycleComposer } from "@/components/cycles/cycle-composer"

export default function NewCyclePage() {
  return (
    <div className="mx-auto w-full max-w-3xl flex-1 px-5 py-12 sm:px-8">
      <header className="mb-8 border-b border-border/60 pb-6">
        <h1 className="font-display text-3xl font-semibold tracking-tight">
          Start a cycle
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Unattended post-training. Leave it running; the cycle page is where it
          reports what it did and why it stopped.
        </p>
      </header>
      <CycleComposer />
    </div>
  )
}
