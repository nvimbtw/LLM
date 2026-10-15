import { Terminal, Beaker } from "lucide-react"
import Link from "next/link"
import { ButtonLink } from "@/components/ui/button-link"
import { cn } from "@/lib/utils"

export interface GymCardProps {
  id: string
  name: string
  description: string
  featured?: boolean
}

/** Presents a gym as an open entry with practice and evaluation links; featured entries span both columns. */
export function GymCard({
  id,
  name,
  description,
  featured = false,
}: GymCardProps) {
  const isBash = id === "bash"
  return (
    <article
      className={cn(
        "flex flex-col gap-4 border-b border-border/60 px-0 py-4 text-sm text-card-foreground",
        featured && "md:col-span-2"
      )}
    >
      <header className="grid auto-rows-min items-start gap-1">
        <div className="flex items-center gap-2">
          {isBash ? (
            <Terminal className="size-4 text-muted-foreground" />
          ) : (
            <Beaker className="size-4 text-muted-foreground" />
          )}
          <h2 className="font-display text-xl leading-snug font-normal">
            {name}
          </h2>
        </div>
        <p className="text-sm text-muted-foreground">{description}</p>
      </header>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <ButtonLink
          href={`/jobs/new/rl?gym=${id}`}
          size="sm"
          variant={featured ? "default" : "outline"}
        >
          Practise
        </ButtonLink>
        <Link
          href={`/gyms/${id}`}
          className="text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground"
        >
          Score once, and past runs
        </Link>
      </div>
    </article>
  )
}
