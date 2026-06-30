import Link from "next/link"

import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import type { Model } from "@/hooks/use-models"
import { ModelGlyph } from "./model-glyph"
import { formatContext, formatParams } from "./format"

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="font-heading text-sm font-medium tabular-nums">{value}</dd>
    </div>
  )
}

export function ModelCard({ model }: { model: Model }) {
  return (
    <Link
      href={`/models/${model.id}`}
      className="rounded-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <Card className="h-full gap-0 pt-0 transition-shadow hover:ring-foreground/20">
        <ModelGlyph tier={model.tier} />
        <CardHeader className="pt-4">
          <CardTitle>{model.name}</CardTitle>
          <CardAction>
            <span className="rounded-full bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground capitalize">
              {model.tier}
            </span>
          </CardAction>
          <CardDescription className="mt-1">{model.description}</CardDescription>
        </CardHeader>
        <CardContent className="pt-4">
          <dl className="grid grid-cols-3 gap-3 border-t pt-4">
            <Stat label="Parameters" value={formatParams(model.params)} />
            <Stat label="Context" value={formatContext(model.contextWindow)} />
            <Stat label="Family" value={model.family} />
          </dl>
        </CardContent>
      </Card>
    </Link>
  )
}
