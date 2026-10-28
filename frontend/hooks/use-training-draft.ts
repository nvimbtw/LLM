"use client"

import * as React from "react"
import { toast } from "@/components/ui/toast"
import {
  parseTrainingDraft,
  loadTrainingDraftAccount,
  readTrainingDraft,
  removeTrainingDraft,
  TRAINING_DRAFT_CHANGED,
  writeTrainingDraft,
} from "@/lib/training-draft"
import type { TrainConfig, TrainingStep } from "@/lib/training"

/** Subscribes sidebar and editor consumers to same-tab saves and cross-tab storage changes, removing both listeners on unmount. */
function subscribe(listener: () => void) {
  window.addEventListener(TRAINING_DRAFT_CHANGED, listener)
  window.addEventListener("storage", listener)
  return () => {
    window.removeEventListener(TRAINING_DRAFT_CHANGED, listener)
    window.removeEventListener("storage", listener)
  }
}

/** Keeps server rendering independent of private browser storage. */
const serverSnapshot = () => null

/** Resolves the captured session's profile identity and observes its local draft. Save/discard remain synchronous browser writes with visible failures. */
export function useTrainingDraft(token: string | null) {
  const [epoch, retry] = React.useReducer((value: number) => value + 1, 0)
  const [account, setAccount] = React.useState<{
    token: string
    accountId: string | null
    error: string | null
  } | null>(null)
  React.useEffect(() => {
    if (!token) return
    const controller = new AbortController()
    void loadTrainingDraftAccount(token, controller.signal)
      .then((accountId) => {
        if (!controller.signal.aborted)
          setAccount({ token, accountId, error: null })
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted)
          setAccount({
            token,
            accountId: null,
            error:
              cause instanceof Error
                ? cause.message
                : "Your account could not be loaded for training drafts.",
          })
      })
    return () => controller.abort()
  }, [token, epoch])
  const currentAccount = account?.token === token ? account : null
  const accountId = currentAccount?.accountId
  const identity = React.useMemo(
    () => (token && accountId ? { token, accountId } : null),
    [token, accountId]
  )
  const snapshot = React.useCallback(
    () => readTrainingDraft(identity),
    [identity]
  )
  const raw = React.useSyncExternalStore(subscribe, snapshot, serverSnapshot)
  const stored = React.useMemo(() => {
    try {
      if (raw instanceof Error) throw raw
      return { draft: raw ? parseTrainingDraft(raw) : null, error: null }
    } catch (cause) {
      return {
        draft: null,
        error:
          cause instanceof Error
            ? cause.message
            : "The training draft could not be read.",
      }
    }
  }, [raw])
  const [writeError, setWriteError] = React.useState<string | null>(null)

  /** Saves a complete editor snapshot immediately so navigation or reload cannot outrun a debounce. */
  const save = (config: TrainConfig, steps: TrainingStep[]) => {
    if (!identity) return
    try {
      writeTrainingDraft(identity, config, steps)
      setWriteError(null)
    } catch (cause) {
      setWriteError(
        cause instanceof Error
          ? cause.message
          : "The training draft could not be saved."
      )
    }
  }

  /** Discards this observed snapshot; failures remain visible and do not pretend the draft was removed. */
  const clear = (expected = raw): boolean => {
    if (!identity) return false
    try {
      const removed = removeTrainingDraft(identity, expected)
      setWriteError(null)
      return removed
    } catch {
      toast.error("The training draft could not be removed from this browser.")
      return false
    }
  }
  return {
    ...stored,
    error: writeError ?? stored.error,
    save,
    clear,
    isLoading: Boolean(token && !currentAccount),
    accountError: currentAccount?.error,
    retry,
  }
}
