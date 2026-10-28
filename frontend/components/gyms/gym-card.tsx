import { Terminal, Beaker } from "lucide-react"
import Link from "next/link"
import { ResourceRow, ResourceTitle } from "@/components/ui/resource-list"

export interface GymCardProps {
  id: string
  name: string
  description: string
}

/** Presents a gym as a single open catalogue entry: its generous title/description link starts practice and a separate link opens evaluation/history. */
export function GymCard({ id, name, description }: GymCardProps) {
  const isBash = id === "bash"
  return (
    <ResourceRow className="flex flex-wrap items-center justify-between gap-x-8 gap-y-3 text-sm">
      <Link
        href={`/gyms/new-gym?gym=${encodeURIComponent(id)}`}
        aria-label={`Practise ${name}`}
        className="group grid min-w-0 flex-1 basis-72 auto-rows-min items-start gap-2 rounded-(--radius-marker) outline-none"
      >
        <div className="flex items-center gap-2">
          {isBash ? (
            <Terminal className="size-4 text-muted-foreground" />
          ) : (
            <Beaker className="size-4 text-muted-foreground" />
          )}
          <ResourceTitle className="group-hover:underline">
            {name}
          </ResourceTitle>
        </div>
        <p className="text-sm text-muted-foreground">{description}</p>
      </Link>
      <div className="flex shrink-0 items-center">
        <Link
          href={`/gyms/sources/${id}`}
          className="text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground"
        >
          Evaluate and view history
        </Link>
      </div>
    </ResourceRow>
  )
}
