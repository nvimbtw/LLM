import { Terminal, Beaker } from "lucide-react"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { ButtonLink } from "@/components/ui/button-link"

export interface GymCardProps {
  id: string
  name: string
  description: string
}

export function GymCard({ id, name, description }: GymCardProps) {
  const isBash = id === "bash"
  return (
    <Card className="transition-shadow hover:shadow-sheet-hover">
      <CardHeader>
        <div className="flex items-center gap-2">
          {isBash ? <Terminal className="size-4 text-muted-foreground" /> : <Beaker className="size-4 text-muted-foreground" />}
          <CardTitle>{name}</CardTitle>
        </div>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <ButtonLink href={`/gyms/${id}`} size="sm" variant="outline">
          Open
        </ButtonLink>
      </CardContent>
    </Card>
  )
}
