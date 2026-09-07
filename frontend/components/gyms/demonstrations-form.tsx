"use client"

import * as React from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Panel } from "@/components/ui/panel"
import { FormError } from "@/components/ui/error-state"
import { createDemonstrations } from "@/lib/gyms"

export interface DemonstrationsFormProps {
  gymId: string
  taskTypes: string[]
}

export function DemonstrationsForm({
  gymId,
  taskTypes,
}: DemonstrationsFormProps) {
  const [name, setName] = React.useState("math-foundations")
  const [count, setCount] = React.useState(1000)
  const [seed, setSeed] = React.useState(0)
  const [busy, setBusy] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [created, setCreated] = React.useState(false)
  const valid =
    name.trim() &&
    taskTypes.length &&
    Number.isInteger(count) &&
    count > 0 &&
    count <= 5000 &&
    Number.isSafeInteger(seed)

  async function submit() {
    setBusy(true)
    setError(null)
    setCreated(false)
    try {
      await createDemonstrations(gymId, taskTypes, name.trim(), count, seed)
      setCreated(true)
      setSeed(seed + count)
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Could not create demonstrations."
      )
    } finally {
      setBusy(false)
    }
  }

  return (
    <Panel
      title="SFT demonstrations"
      description="Generate checked examples for the selected task types. Choose easy tasks first if the model rarely answers correctly."
    >
      <div className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-3">
          <label className="space-y-1 text-sm">
            Dataset name
            <Input value={name} onChange={(e) => setName(e.target.value)} />
          </label>
          <label className="space-y-1 text-sm">
            Examples
            <Input
              type="number"
              min={1}
              max={5000}
              value={count}
              onChange={(e) => setCount(e.target.valueAsNumber)}
            />
          </label>
          <label className="space-y-1 text-sm">
            First seed
            <Input
              type="number"
              value={seed}
              onChange={(e) => setSeed(e.target.valueAsNumber)}
            />
          </label>
        </div>
        <p className="text-xs text-muted-foreground">
          An existing dataset name appends examples. This creates data; it does
          not start training.
        </p>
        {error && <FormError>{error}</FormError>}
        <Button variant="outline" disabled={busy || !valid} onClick={submit}>
          {busy ? "Generating…" : "Create demonstrations"}
        </Button>
        {created && (
          <p className="text-sm">
            Examples saved. Select this dataset in{" "}
            <Link href="/jobs/new/train" className="underline">
              a fine-tuning run
            </Link>
            .
          </p>
        )}
      </div>
    </Panel>
  )
}
