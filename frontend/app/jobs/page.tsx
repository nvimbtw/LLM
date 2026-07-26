import { JobsView } from "@/components/jobs/jobs-view"

export default function JobsPage() {
  return (
    <div className="mx-auto w-full max-w-5xl flex-1 px-5 py-12 sm:px-8">
      <header className="mb-8 border-b border-border/60 pb-6">
        <h1 className="font-display text-3xl font-semibold tracking-tight">Jobs</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          One GPU, one line. Training runs, benchmarks and chat turns queue here and
          execute in order — and the queue survives a restart, so what you leave
          waiting is still waiting.
        </p>
      </header>
      <JobsView />
    </div>
  )
}
