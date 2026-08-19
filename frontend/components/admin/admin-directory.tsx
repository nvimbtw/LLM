import Link from "next/link"
import {
  ArrowRight,
  Brain,
  Database,
  Gauge,
  Loader,
  MessageSquareCode,
  Type,
} from "lucide-react"

/**
 * Everything an admin can reach that is not the queue.
 *
 * These used to be six rows in the sidebar, which put pages nobody visits twice a
 * week — a throbber gallery, a markup fixture — permanently beside Chat and
 * Conversations. A directory can afford a sentence per destination; a nav row
 * cannot, and a nav row with only a two-word label leaves the reader to guess
 * whether "Markup" is a setting, a corpus or a page.
 *
 * Two bands in one frame, the same move the queue makes: the split is what puts
 * work on the GPU versus what only shows how the interface behaves, and a border
 * between two *sections* would make that distinction louder than the queue above
 * it — which is the page.
 */
interface Destination {
  title: string
  href: string
  icon: React.ComponentType<{ className?: string }>
  blurb: string
}

const WORKBENCH: Destination[] = [
  {
    title: "Data",
    href: "/data",
    icon: Database,
    blurb:
      "Pull a dataset from Hugging Face, read what actually arrived, and clean it — watching each step on a handful of documents first.",
  },
  {
    title: "Tokenizers",
    href: "/tokenizers",
    icon: Type,
    blurb:
      "Learn a vocabulary from the corpora you pick, then keep it. Every run sharing one shares the packed corpus underneath.",
  },
  {
    title: "Train",
    href: "/jobs/new/train",
    icon: Brain,
    blurb:
      "Compose a training run — architecture, corpus, and the dozen numbers that make it a different model. Submitting queues it.",
  },
  {
    title: "Benchmark",
    href: "/benchmark",
    icon: Gauge,
    blurb:
      "Held-out perplexity, minimal-pair grammar and surface metrics. History and analysis; runs start from the queue.",
  },
]

const INTERFACE: Destination[] = [
  {
    title: "Throbber",
    href: "/throbber",
    icon: Loader,
    blurb:
      "Every state of the figure side by side, so a change to one can be judged against the other seven.",
  },
  {
    title: "Markup",
    href: "/markup",
    icon: MessageSquareCode,
    blurb:
      "Every renderable special token in one fixture conversation, through the real chat bubble.",
  },
]

export function AdminDirectory() {
  const total = WORKBENCH.length + INTERFACE.length

  return (
    <section className="overflow-hidden rounded-lg border border-border bg-card">
      <header className="border-b border-border px-5 py-4">
        <h2 className="text-sm font-semibold tracking-tight">Elsewhere</h2>
        <p className="mt-0.5 text-xs font-medium tracking-[0.12em] text-muted-foreground uppercase">
          {total} pages
        </p>
      </header>

      <Band label="Workbench" destinations={WORKBENCH} />
      {/* Nothing here touches the GPU or the database — they render fixtures. The
          rule says so without a paragraph explaining it. */}
      <Band
        label="Interface"
        destinations={INTERFACE}
        className="border-t border-border/60"
      />
    </section>
  )
}

function Band({
  label,
  destinations,
  className,
}: {
  label: string
  destinations: Destination[]
  className?: string
}) {
  return (
    <div className={className}>
      <p className="px-5 pt-4 pb-1 text-[0.6875rem] font-medium tracking-[0.16em] text-muted-foreground/70 uppercase">
        {label}
      </p>
      <ul>
        {destinations.map((destination) => (
          <li
            key={destination.href}
            className="border-b border-border/40 last:border-0"
          >
            <Link
              href={destination.href}
              className="group flex items-start gap-3 px-5 py-3 transition-colors hover:bg-accent/40"
            >
              <destination.icon className="mt-0.5 size-4 shrink-0 text-muted-foreground/70 transition-colors group-hover:text-foreground" />
              <div className="min-w-0 flex-1">
                <span className="block text-sm font-medium">
                  {destination.title}
                </span>
                {/* Capped at the reading measure rather than the board's width: a
                    sentence set across 110rem is a sentence nobody finishes. */}
                <span className="mt-0.5 block max-w-2xl text-xs leading-relaxed text-muted-foreground">
                  {destination.blurb}
                </span>
              </div>
              <ArrowRight
                className="mt-0.5 size-3.5 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100"
                aria-hidden
              />
            </Link>
          </li>
        ))}
      </ul>
    </div>
  )
}
