"use client"

import * as React from "react"

import { useSessionToken } from "@/hooks/use-auth"
import { getToken } from "@/lib/api"
import { createProcessor } from "@/lib/data"

const STARTER_SOURCE = `from processors.helpers import Ctx, Param, processor


@processor(name="my-processor", title="My processor", kinds=["pretrain", "sft"], params=[])
def run(text: str, p: dict, ctx: Ctx) -> str | None:
    """Transform text, or return None to drop its document."""
    return text
`

/** Owns a new processor's source draft and a cancellable create request for one session. */
export function useNewProcessor(onCreated: (name: string) => void) {
  const token = useSessionToken()
  const [source, setSource] = React.useState(STARTER_SOURCE)
  const [submitting, setSubmitting] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const pending = React.useRef<AbortController | null>(null)
  React.useEffect(() => () => pending.current?.abort(), [])

  /** Creates a valid first version and ignores completions after session replacement or disposal. */
  async function submit() {
    if (!token || getToken() !== token || submitting || pending.current) return
    const controller = new AbortController()
    pending.current = controller
    setSubmitting(true)
    setError(null)
    try {
      const created = await createProcessor(source, controller.signal)
      if (!controller.signal.aborted && getToken() === token)
        onCreated(created.name)
    } catch (failure) {
      if (!controller.signal.aborted && getToken() === token)
        setError(
          failure instanceof Error
            ? failure.message
            : "Processor was not created"
        )
    } finally {
      if (pending.current === controller) pending.current = null
      if (!controller.signal.aborted && getToken() === token)
        setSubmitting(false)
    }
  }

  return { source, setSource, submitting, error, submit }
}
