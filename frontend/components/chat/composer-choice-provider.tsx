"use client"

import * as React from "react"

import type { ThinkBudget } from "./budget-selector"

/**
 * What the composer is set to send with — the model, and how hard it should
 * think.
 *
 * It lives above the router rather than inside `ChatComposer` because the
 * first message of a conversation is sent from one composer (the new-chat
 * screen) and every message after it from another (the thread), and those are
 * two different mounts. Held here, the choice survives that navigation, which
 * is the only reason this is a context and not local state: nothing else reads
 * it.
 *
 * Deliberately not persisted. The budget is a property of the question you are
 * about to ask rather than a preference (see `budget-selector.tsx`), and the
 * model list can change under a stored id between sessions; both are worth
 * carrying across a navigation, neither across a reload.
 */
interface ComposerChoiceValue {
  /** Empty until the model list has loaded — see `ModelSelector`. */
  model: string
  setModel: (id: string) => void
  budget: ThinkBudget
  setBudget: (budget: ThinkBudget) => void
}

const ComposerChoiceContext = React.createContext<ComposerChoiceValue | null>(
  null
)

export function ComposerChoiceProvider({
  children,
}: {
  children: React.ReactNode
}) {
  const [model, setModel] = React.useState("")
  const [budget, setBudget] = React.useState<ThinkBudget>("auto")

  const value = React.useMemo(
    () => ({ model, setModel, budget, setBudget }),
    [model, budget]
  )

  return (
    <ComposerChoiceContext.Provider value={value}>
      {children}
    </ComposerChoiceContext.Provider>
  )
}

export function useComposerChoice() {
  const ctx = React.useContext(ComposerChoiceContext)
  if (!ctx) {
    throw new Error(
      "useComposerChoice must be used within a ComposerChoiceProvider"
    )
  }
  return ctx
}
