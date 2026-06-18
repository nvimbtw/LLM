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

function formatParams(billions: number) {
  if (!billions) return "Undisclosed"
  if (billions >= 1000) {
    const t = billions / 1000
    return `${t % 1 ? t.toFixed(1) : t}T`
  }
  return `${billions}B`
}

function formatContext(tokens: number) {
  if (tokens >= 1_000_000) {
    const m = tokens / 1_000_000
    return `${m % 1 ? m.toFixed(1) : m}M`
  }
  if (tokens >= 1000) return `${Math.round(tokens / 1000)}K`
  return `${tokens}`
}

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
    <Card className="gap-0 pt-0 transition-shadow hover:ring-foreground/20">
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
  )
}
