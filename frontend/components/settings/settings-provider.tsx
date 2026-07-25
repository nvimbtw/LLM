"use client"

import * as React from "react"

import { DEFAULT_SETTINGS, type Settings } from "@/lib/settings"

const STORAGE_KEY = "llm.settings"

interface SettingsContextValue {
  settings: Settings
  setSetting: <K extends keyof Settings>(key: K, value: Settings[K]) => void
  resetSettings: () => void
  // The command modal's open state lives here so Ctrl+K, the sidebar, and slash
  // commands (`/settings`) can all drive the same dialog.
  commandOpen: boolean
  setCommandOpen: (open: boolean) => void
}

const SettingsContext = React.createContext<SettingsContextValue | null>(null)

function loadSettings(): Settings {
  if (typeof window === "undefined") return DEFAULT_SETTINGS
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return DEFAULT_SETTINGS
    // Merge over defaults so a newly-added setting picks up its default rather
    // than being undefined for anyone with older persisted state.
    return { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<Settings>) }
  } catch {
    return DEFAULT_SETTINGS
  }
}

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  // Start from defaults on both server and first client render (no hydration
  // mismatch); hydrate from localStorage in an effect right after mount.
  const [settings, setSettings] = React.useState<Settings>(DEFAULT_SETTINGS)
  const [commandOpen, setCommandOpen] = React.useState(false)

  React.useEffect(() => {
    // One-time hydration from localStorage after mount, kept out of render so
    // server and first client render match (SSR-safe).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSettings(loadSettings())
  }, [])

  const setSetting = React.useCallback(
    <K extends keyof Settings>(key: K, value: Settings[K]) => {
      setSettings((prev) => {
        const next = { ...prev, [key]: value }
        try {
          window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
        } catch {
          // Storage unavailable (private mode / quota) — keep it in memory.
        }
        return next
      })
    },
    []
  )

  const resetSettings = React.useCallback(() => {
    setSettings(DEFAULT_SETTINGS)
    try {
      window.localStorage.removeItem(STORAGE_KEY)
    } catch {
      // ignore
    }
  }, [])

  const value = React.useMemo(
    () => ({ settings, setSetting, resetSettings, commandOpen, setCommandOpen }),
    [settings, setSetting, resetSettings, commandOpen]
  )

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>
}

export function useSettings(): SettingsContextValue {
  const ctx = React.useContext(SettingsContext)
  if (!ctx) throw new Error("useSettings must be used within a SettingsProvider")
  return ctx
}
