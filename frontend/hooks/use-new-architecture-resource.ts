"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import type { ArchitectureKind } from "./use-architecture-catalog"
import { architectures } from "@/lib/architectures"
import { blocks } from "@/lib/blocks"
import { ArchitectureApiError } from "@/lib/architecture/http"
import { ArchitectureReadOwner } from "@/lib/architecture/read-owner"
import { readCopiedDraft } from "@/lib/architecture/copy"
import { encodeArchitecture } from "@/lib/architecture/decoder"

/** Creates one draft or explicitly copies a pinned revision, preserving ambiguous network outcomes without automatic duplicate creation. */
export function useNewArchitectureResource(
  kind: ArchitectureKind,
  copy: string | undefined,
  revision: string | undefined,
  onCreated: (id: string) => void
) {
  const token = useSessionToken()
  const [name, setName] = React.useState("")
  const [description, setDescription] = React.useState("")
  const [text, setText] = React.useState("")
  const [copyReady, setCopyReady] = React.useState(!copy && !revision)
  const [error, setError] = React.useState<string | null>(null)
  const [submitting, setSubmitting] = React.useState(false)
  const [uncertain, setUncertain] = React.useState(false)
  const active = React.useRef(false)
  const pending = React.useRef<AbortController | null>(null)
  React.useEffect(() => {
    active.current = true
    const owner = new ArchitectureReadOwner({
      token: token ?? "",
      resource: `${kind}:copy:${copy}:${revision}`,
    })
    if (token && (copy || revision)) {
      owner
        .read(async (signal) => {
          if (!copy || !revision)
            throw new Error(
              "Copy requires an exact resource and accepted revision."
            )
          return readCopiedDraft(kind, copy, revision, { token, signal })
        })
        .then((value) => {
          if (!value || !active.current) return
          setName(value.name)
          setDescription(value.description ?? "")
          setText(value.source)
          setCopyReady(true)
        })
        .catch((error: unknown) => {
          if (active.current)
            setError(
              error instanceof Error
                ? error.message
                : "Selected revision could not be copied."
            )
        })
    }
    return () => {
      active.current = false
      owner.dispose()
      pending.current?.abort()
    }
  }, [token, kind, copy, revision])

  /** Submits one captured creation; uncertain transport failure requires checking the directory before another creation attempt. */
  async function submit() {
    if (!token || !copyReady || uncertain || pending.current || !name.trim())
      return
    try {
      if (kind === "block")
        encodeArchitecture("BlockCreate", { name, description, source: text })
      else
        encodeArchitecture("ArchitectureCreate", {
          name,
          description,
          yaml: text,
        })
    } catch (failure) {
      setError(
        failure instanceof Error
          ? failure.message
          : "Check the draft fields before creating it."
      )
      return
    }
    const controller = new AbortController()
    pending.current = controller
    setSubmitting(true)
    setError(null)
    try {
      const request = { token, signal: controller.signal }
      const result =
        kind === "block"
          ? await blocks.create({ name, description, source: text }, request)
          : await architectures.create(
              { name, description, yaml: text },
              request
            )
      if (active.current) onCreated(result.resource_id)
    } catch (failure) {
      if (!active.current) return
      const rejected =
        failure instanceof ArchitectureApiError &&
        failure.status >= 400 &&
        failure.status < 500
      setUncertain(!rejected)
      setError(
        rejected
          ? failure.message
          : "Creation status is unknown. Keep your source and check the directory before creating another resource."
      )
    } finally {
      if (pending.current === controller) pending.current = null
      if (active.current) setSubmitting(false)
    }
  }

  return {
    name,
    setName,
    description,
    setDescription,
    text,
    setText,
    copyReady,
    error,
    submitting,
    uncertain,
    submit,
  }
}
