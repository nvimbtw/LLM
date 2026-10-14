"use client"

import * as React from "react"
import type { ThinkBudget } from "@/lib/think-budget"
import { getToken } from "@/lib/api"
import { useAuth, useSessionToken } from "@/hooks/use-auth"
import {
  DEFAULT_COMPOSER_CHOICE,
  loadComposerChoice,
  persistComposerChoice,
  type ComposerChoice,
} from "@/lib/composer-choice"

interface ComposerChoiceValue {
  isLoading: boolean
  model: string
  setModel: (id: string) => void
  budget: ThinkBudget
  setBudget: (budget: ThinkBudget) => void
}

interface ChoiceState {
  token: string
  choice: ComposerChoice
  loaded: boolean
}
const ComposerChoiceContext = React.createContext<ComposerChoiceValue | null>(
  null
)

/** Owns the active account's model/budget readiness and immediate writes, preserving selections across composer navigation. */
export function ComposerChoiceProvider({
  children,
}: {
  children: React.ReactNode
}) {
  const token = useSessionToken()
  const { isLoading: authLoading } = useAuth()
  const [state, setState] = React.useState<ChoiceState | null>(null)
  const current = state?.token === token ? state : null
  const choice = current?.choice ?? DEFAULT_COMPOSER_CHOICE

  React.useEffect(() => {
    if (!token) return
    const controller = new AbortController()
    loadComposerChoice(token, controller.signal)
      .then((loadedChoice) => {
        if (!controller.signal.aborted && getToken() === token)
          setState({ token, choice: loadedChoice, loaded: true })
      })
      .catch(() => {
        if (!controller.signal.aborted && getToken() === token)
          setState({ token, choice: DEFAULT_COMPOSER_CHOICE, loaded: true })
      })
    return () => controller.abort()
  }, [token])

  /** Applies an explicit local pick and writes only after this same account's choice has settled. */
  const change = React.useCallback(
    (next: ComposerChoice): void => {
      if (!token || getToken() !== token) return
      const loaded = current?.loaded ?? false
      setState({ token, choice: next, loaded })
      if (loaded) persistComposerChoice(token, next)
    },
    [token, current]
  )

  /** Selects a model or checkpoint without changing the selected thinking budget. */
  const setModel = React.useCallback(
    (model: string): void => change({ ...choice, model }),
    [change, choice]
  )
  /** Selects a thinking budget without changing the selected model or checkpoint. */
  const setBudget = React.useCallback(
    (budget: ThinkBudget): void => change({ ...choice, budget }),
    [change, choice]
  )

  return (
    <ComposerChoiceContext.Provider
      value={{
        isLoading: authLoading || (Boolean(token) && !current?.loaded),
        model: choice.model,
        budget: choice.budget,
        setModel,
        setBudget,
      }}
    >
      {children}
    </ComposerChoiceContext.Provider>
  )
}

/** Returns the current account's real composer choice and readiness; requires its provider. */
export function useComposerChoice(): ComposerChoiceValue {
  const value = React.useContext(ComposerChoiceContext)
  if (!value)
    throw new Error(
      "useComposerChoice must be used within a ComposerChoiceProvider"
    )
  return value
}
