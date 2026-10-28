import Link from "next/link"
import { ChevronDown } from "lucide-react"
import { AdminSectionHeader } from "./admin-section-header"

interface Destination {
  title: string
  href: string
  legacy?: boolean
}
interface DestinationGroup {
  title: string
  destinations: Destination[]
}

const WORKBENCH: DestinationGroup[] = [
  {
    title: "Data",
    destinations: [
      { title: "Processors", href: "/processors" },
      { title: "Datasets", href: "/data" },
    ],
  },
  {
    title: "Architectures",
    destinations: [
      { title: "Blocks", href: "/arch/blocks" },
      { title: "Architectures", href: "/arch" },
    ],
  },
  {
    title: "Train",
    destinations: [
      { title: "Tokenizers", href: "/tokenizers" },
      { title: "Pre-train", href: "/train/new-train?mode=pretrain" },
      { title: "Fine-tune", href: "/train/new-train?mode=sft" },
      { title: "Gyms", href: "/gyms" },
    ],
  },
  {
    title: "Bench",
    destinations: [
      { title: "Language", href: "/bench?suite=language" },
      { title: "Language v3", href: "/bench?suite=language_v3", legacy: true },
      { title: "Markup", href: "/bench?suite=markup", legacy: true },
      { title: "Assistant", href: "/bench?suite=assistant", legacy: true },
    ],
  },
]
const INTERFACE: DestinationGroup = {
  title: "Interface",
  destinations: [
    { title: "Throbber", href: "/throbber" },
    { title: "Markup", href: "/markup" },
    { title: "Docs", href: "/docs" },
  ],
}

/** Groups Workbench destinations beside the queue; native disclosures retain keyboard and narrow-screen access. */
export function AdminDirectory() {
  return (
    <section aria-label="Workbench">
      <AdminSectionHeader title="Workbench" />
      <div className="space-y-5">
        {WORKBENCH.map((group) => (
          <DirectoryGroup key={group.title} group={group} open />
        ))}
        <DirectoryGroup group={INTERFACE} />
      </div>
    </section>
  )
}

/** Renders a collapsible destination group, with legacy suites explicitly distinguished from new execution. */
function DirectoryGroup({
  group,
  open = false,
}: {
  group: DestinationGroup
  open?: boolean
}) {
  return (
    <details open={open} className="group/directory">
      <summary className="flex min-h-9 cursor-pointer list-none items-center justify-between gap-3 text-sm font-medium outline-none focus-visible:ring-2 focus-visible:ring-ring [&::-webkit-details-marker]:hidden">
        {group.title}
        <ChevronDown
          aria-hidden
          className="size-3.5 text-muted-foreground transition-transform group-open/directory:rotate-180"
        />
      </summary>
      <ul className="mt-1 space-y-1">
        {group.destinations.map((destination) => (
          <li key={destination.href}>
            <Link
              href={destination.href}
              className="flex min-h-9 items-center justify-between gap-3 py-1 pl-3 text-sm text-muted-foreground outline-none hover:text-primary focus-visible:ring-2 focus-visible:ring-ring"
            >
              <span>{destination.title}</span>
              {destination.legacy && <span className="text-xs">Legacy</span>}
            </Link>
          </li>
        ))}
      </ul>
    </details>
  )
}
