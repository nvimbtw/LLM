"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { getToken } from "@/lib/api"
import { usePageDraft } from "./use-page-draft"
import { readDraftStrings } from "@/lib/page-draft"
import {
  getTokenizer,
  editTokenizer,
  deleteTokenizer,
  type TokenizerDetail,
  type TokenizerFields,
} from "@/lib/tokenizers"

/** Restores exactly the vocabulary's editable text fields, preserving unfinished names and multiline descriptions. */
function tokenizerDraft(value: unknown): TokenizerFields {
  return readDraftStrings(["name", "description"], value)
}

/** Owns one mounted session/resource's metadata, draft and writes; stale completions cannot mutate or navigate its replacement. */
export function useTokenizer(id: string, onDeleted: () => void) {
  const token = useSessionToken()
  const [tokenizer, setTokenizer] = React.useState<TokenizerDetail | null>(null)
  const draft = usePageDraft(
    `tokenizer:${id}`,
    { name: tokenizer?.name ?? "", description: tokenizer?.description ?? "" },
    tokenizerDraft,
    Boolean(tokenizer)
  )
  const fields = draft.fields
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  const [failure, setFailure] = React.useState<string | null>(null)
  const [busy, setBusy] = React.useState(false)
  const pending = React.useRef(false)
  const active = React.useRef(false)

  React.useEffect(() => {
    active.current = true
    let current = true
    const controller = new AbortController()
    getTokenizer(id, controller.signal)
      .then((next) => {
        if (!current || getToken() !== token) return
        setTokenizer(next)
      })
      .catch((cause: unknown) => {
        if (current && getToken() === token)
          setError(
            cause instanceof Error
              ? cause.message
              : "The vocabulary didn't load."
          )
      })
      .finally(() => {
        if (current && getToken() === token) setLoading(false)
      })
    return () => {
      current = false
      active.current = false
      controller.abort()
    }
  }, [id, token])

  /** Changes a local draft field; persistence occurs only on Save. */
  const change = (field: keyof TokenizerFields, value: string) =>
    draft.change({ ...fields, [field]: value })

  /** Restores the stored display fields and clears the last submission failure. */
  const cancel = () => {
    if (!tokenizer || pending.current) return
    draft.reset()
    setFailure(null)
  }

  /** Persists validated fields, preserving the draft when validation or the server rejects it. */
  const save = async () => {
    if (!token || pending.current || getToken() !== token) return
    pending.current = true
    setBusy(true)
    setFailure(null)
    try {
      const saved = await editTokenizer(token, id, fields)
      if (!active.current || getToken() !== token) return
      draft.change({ name: saved.name, description: saved.description })
      draft.forget()
      setTokenizer(saved)
    } catch (cause) {
      if (active.current && getToken() === token)
        setFailure(
          cause instanceof Error
            ? cause.message
            : "The vocabulary could not be saved."
        )
    } finally {
      pending.current = false
      if (active.current && getToken() === token) setBusy(false)
    }
  }

  /** Deletes only after confirmation and navigates only while this session/resource remains mounted. */
  const remove = async () => {
    if (!token || pending.current || getToken() !== token) return
    pending.current = true
    setBusy(true)
    setFailure(null)
    try {
      await deleteTokenizer(id)
      if (active.current && getToken() === token) onDeleted()
    } catch (cause) {
      if (active.current && getToken() === token)
        setFailure(
          cause instanceof Error
            ? cause.message
            : "The vocabulary could not be deleted."
        )
      throw cause
    } finally {
      pending.current = false
      if (active.current && getToken() === token) setBusy(false)
    }
  }

  return {
    tokenizer,
    fields,
    loading,
    error,
    failure: failure ?? draft.persistenceError,
    busy,
    change,
    cancel,
    save,
    remove,
    changed:
      tokenizer !== null &&
      (fields.name !== tokenizer.name ||
        fields.description !== tokenizer.description),
  }
}
