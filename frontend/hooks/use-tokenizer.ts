"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { getToken } from "@/lib/api"
import {
  getTokenizer,
  editTokenizer,
  deleteTokenizer,
  type TokenizerDetail,
  type TokenizerFields,
} from "@/lib/tokenizers"

/** Owns one mounted session/resource's metadata, draft and writes; stale completions cannot mutate or navigate its replacement. */
export function useTokenizer(id: string, onDeleted: () => void) {
  const token = useSessionToken()
  const [tokenizer, setTokenizer] = React.useState<TokenizerDetail | null>(null)
  const [fields, setFields] = React.useState<TokenizerFields>({
    name: "",
    description: "",
  })
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
        setFields({ name: next.name, description: next.description })
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
    setFields((previous) => ({ ...previous, [field]: value }))

  /** Restores the stored display fields and clears the last submission failure. */
  const cancel = () => {
    if (!tokenizer || pending.current) return
    setFields({ name: tokenizer.name, description: tokenizer.description })
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
      setTokenizer(saved)
      setFields({ name: saved.name, description: saved.description })
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
    failure,
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
