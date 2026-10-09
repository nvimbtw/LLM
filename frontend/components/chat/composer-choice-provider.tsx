"use client"

import * as React from "react"

import type { ThinkBudget } from "@/lib/think-budget"
import { apiFetch } from "@/lib/api"
import { useAuth } from "@/hooks/use-auth"

const DEFAULT_BUDGET: ThinkBudget = "auto"

/**
 * What the composer is set to send with — the model, and how hard it should
 * think.
 *
 * It lives above the router rather than inside `ChatComposer` because the
 * first message of a conversation is sent from one composer (the new-chat
 * screen) and every message after it from another (the thread), and those are
 * two different mounts. Held here, the choice survives that navigation.
 *
 * Persisted to the account (`GET`/`PUT /user/me/composer-choice`) as the
 * starting point for the *next* session — not folded into `lib/settings.ts`,
 * because it isn't rendered in the ⌘K modal and doesn't share that data's
 * write rhythm (see the provider body). A stored model id can go stale
 * between sessions if the model is deleted or unpublished; `ComposerPicker`
 * already corrects an id that isn't in the loaded list, so that's handled
 * once, at the one place that has the list.
 */
interface ComposerChoiceValue {
  isLoading: boolean
  /** Empty until the model list has loaded, or no model was ever chosen — see
   *  `ComposerPicker`. */
  model: string
  setModel: (id: string) => void
  budget: ThinkBudget
  setBudget: (budget: ThinkBudget) => void
}

const ComposerChoiceContext = React.createContext<ComposerChoiceValue | null>(
  null
)

interface Choice {
  model: string
  budget: ThinkBudget
}

function persist(choice: Choice) {
  void apiFetch("/user/me/composer-choice", {
    method: "PUT",
    body: JSON.stringify({ composer_choice: choice }),
  }).catch(() => {
    // A dropped write leaves the account's copy stale, not this tab's — the
    // next pick, or the next load, tries again.
  })
}

export function ComposerChoiceProvider({
  children,
}: {
  children: React.ReactNode
}) {
  const { isAuthenticated, isLoading: authLoading } = useAuth()
  const [model, setModelState] = React.useState("")
  const [budget, setBudgetState] = React.useState<ThinkBudget>(DEFAULT_BUDGET)
  const [isLoaded, setIsLoaded] = React.useState(false)
  const [loadedFor, setLoadedFor] = React.useState(isAuthenticated)
  if (loadedFor !== isAuthenticated) {
    setLoadedFor(isAuthenticated)
    setIsLoaded(false)
  }

  // Whether the account's own copy has landed — see `SettingsProvider`, same
  // reasoning: a pick made in the gap before it does must not PUT a
  // still-default choice over whatever was already saved.
  const loaded = React.useRef(false)

  // Unlike settings, a model or budget pick is a single discrete action, not
  // a value dragged or typed through many intermediate states — so each
  // write goes straight out, no debounce to hold.
  const write = React.useCallback((next: Choice) => {
    if (loaded.current) persist(next)
  }, [])

  React.useEffect(() => {
    loaded.current = false
    if (!isAuthenticated) return
    let active = true
    apiFetch("/user/me/composer-choice")
      .then(async (response) => {
        if (!response.ok) throw new Error("Unable to load composer choice.")
        return (await response.json()) as { composer_choice: Partial<Choice> }
      })
      .then((data) => {
        if (!active) return
        if (data.composer_choice.model)
          setModelState(data.composer_choice.model)
        if (data.composer_choice.budget)
          setBudgetState(data.composer_choice.budget)
      })
      .catch(() => {
        // Stay on defaults; the next pick still saves normally.
      })
      .finally(() => {
        if (active) {
          loaded.current = true
          setIsLoaded(true)
        }
      })
    return () => {
      active = false
    }
  }, [isAuthenticated])

  const setModel = React.useCallback(
    (id: string) => {
      setModelState(id)
      write({ model: id, budget })
    },
    [budget, write]
  )

  const setBudget = React.useCallback(
    (next: ThinkBudget) => {
      setBudgetState(next)
      write({ model, budget: next })
    },
    [model, write]
  )

  const value = React.useMemo(
    () => ({
      isLoading: authLoading || (isAuthenticated && !isLoaded),
      // Derived, not stored: logging out must not leave the previous
      // session's choice on screen — same reasoning as `ChatsProvider` and
      // `SettingsProvider`.
      model: isAuthenticated ? model : "",
      setModel,
      budget: isAuthenticated ? budget : DEFAULT_BUDGET,
      setBudget,
    }),
    [authLoading, isAuthenticated, isLoaded, model, setModel, budget, setBudget]
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
