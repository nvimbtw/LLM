"use client"

import * as React from "react"

import { Button } from "@/components/ui/button"
import { FormError } from "@/components/ui/error-state"
import { Input } from "@/components/ui/input"
import { stepGymEpisode, type GymStep } from "@/lib/gyms"

/**
 * A command typed by a person, in an episode's own workspace.
 *
 * Here because a task no model can solve is a task to look at by hand, and this
 * is the only way to look: the filesystem lives in the sandbox, keyed by the
 * episode, and nothing else can reach it.
 *
 * It is recorded as a step like any other — the gyms engine does not know who
 * typed it — so this is deliberately absent once an episode is graded, where an
 * extra step would be appended to a trajectory that has already been scored.
 */
export function CommandBox({ episodeId }: { episodeId: string }) {
  const [command, setCommand] = React.useState("")
  const [ran, setRan] = React.useState<GymStep | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const [busy, setBusy] = React.useState(false)

  const submit = async () => {
    if (!command.trim() || busy) return
    setBusy(true)
    setError(null)
    try {
      setRan(await stepGymEpisode(episodeId, command))
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.")
    } finally {
      setBusy(false)
    }
  }

  const output = [ran?.stdout, ran?.stderr].filter((s) => s?.trim()).join("\n")

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <Input
          value={command}
          onChange={(e) => setCommand(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder="grep -rl NEEDLE ."
          className="font-mono text-xs"
        />
        <Button variant="outline" onClick={submit} disabled={busy}>
          {busy ? "Running…" : "Run"}
        </Button>
      </div>
      {error && <FormError>{error}</FormError>}
      {ran && (
        <pre className="overflow-x-auto rounded-(--radius-sheet) bg-muted px-3 py-2 font-mono text-xs">
          {output || `exit ${ran.exit_code}`}
        </pre>
      )}
    </div>
  )
}
