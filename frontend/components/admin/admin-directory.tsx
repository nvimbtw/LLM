import Link from "next/link"
import { AdminSectionHeader } from "./admin-section-header"
import {
  ArrowRight,
  Brain,
  Database,
  Dumbbell,
  Gauge,
  Loader,
  MessageSquareCode,
  Type,
  BookText,
} from "lucide-react"

interface Destination {
  title: string
  href: string
  icon: React.ComponentType<{ className?: string }>
}

const WORKBENCH: Destination[] = [
  {
    title: "Data",
    href: "/data",
    icon: Database,
  },
  {
    title: "Tokenizers",
    href: "/tokenizers",
    icon: Type,
  },
  {
    title: "Train",
    href: "/jobs/new-train",
    icon: Brain,
  },
  {
    title: "Benchmark",
    href: "/benchmark",
    icon: Gauge,
  },
  {
    title: "Gyms",
    href: "/gyms",
    icon: Dumbbell,
  },
]

const INTERFACE: Destination[] = [
  {
    title: "Throbber",
    href: "/throbber",
    icon: Loader,
  },
  {
    title: "Markup",
    href: "/markup",
    icon: MessageSquareCode,
  },
  {
    title: "Docs",
    href: "/docs",
    icon: BookText,
  },
]

export function AdminDirectory() {
  const total = WORKBENCH.length + INTERFACE.length

  return (
    <section>
      <AdminSectionHeader title="Elsewhere" meta={`${total} pages`} />

      <Band label="Workbench" destinations={WORKBENCH} />
      <Band
        label="Interface"
        destinations={INTERFACE}
        className="mt-4 border-t border-border/60 pt-4"
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
      <p className="mb-2 eyebrow text-muted-foreground">{label}</p>
      <ul className="-mx-3 space-y-1">
        {destinations.map((destination) => (
          <li key={destination.href}>
            <Link
              href={destination.href}
              className="group flex items-center gap-3 rounded-(--radius-control-sm) px-3 py-2 transition-colors outline-none hover:bg-muted/50 focus-visible:bg-muted/50 focus-visible:ring-2 focus-visible:ring-ring"
            >
              <destination.icon
                className="size-4 shrink-0 text-muted-foreground transition-colors group-hover:text-foreground group-focus-visible:text-foreground"
                aria-hidden
              />
              <div className="min-w-0 flex-1">
                <span className="block text-sm">{destination.title}</span>
              </div>
              <ArrowRight
                className="size-3.5 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100"
                aria-hidden
              />
            </Link>
          </li>
        ))}
      </ul>
    </div>
  )
}
