import { Terminal, Beaker } from "lucide-react"
import Link from "next/link"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { ButtonLink } from "@/components/ui/button-link"
import { cn } from "@/lib/utils"

export interface GymCardProps {
  id: string
  name: string
  description: string
  featured?: boolean
}

export function GymCard({ id, name, description, featured = false }: GymCardProps) {
  const isBash = id === "bash"
  return (
    <Card className={cn("transition-shadow hover:shadow-sheet-hover", featured && "md:col-span-2 ring-primary/35")}>
      <CardHeader>
        <div className="flex items-center gap-2">
          {isBash ? <Terminal className="size-4 text-muted-foreground" /> : <Beaker className="size-4 text-muted-foreground" />}
          <CardTitle className={featured ? "text-xl" : undefined}>{name}</CardTitle>
        </div>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        {/* Practising is what the catalogue is for, so it is one click from
            here and arrives with this gym already chosen. The other door —
            one evaluation, and the board of the ones already run — is still a
            door, just the quieter one. */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <ButtonLink href={`/jobs/new/rl?gym=${id}`} size="sm" variant={featured ? "default" : "outline"}>
            Practise
          </ButtonLink>
          <Link
            href={`/gyms/${id}`}
            className="text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground"
          >
            Score once, and past runs
          </Link>
        </div>
      </CardContent>
    </Card>
  )
}
