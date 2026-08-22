"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"
import { useAuth } from "@/hooks/use-auth"
import { DEFAULT_SETTINGS, type Settings } from "@/lib/settings"

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

/**
 * How long a change sits before it's written back. A slider dragged across
 * its range or a system prompt typed character by character would otherwise
 * be one PUT per tick; this holds the last value of a burst and sends that.
 */
const WRITE_DEBOUNCE_MS = 500

function persist(settings: Settings) {
  void apiFetch("/user/me/settings", {
    method: "PUT",
    body: JSON.stringify({ settings }),
  }).catch(() => {
    // A dropped write leaves the account's copy stale, not this tab's — the
    // next change, or the next load, tries again.
  })
}

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth()
  const [settings, setSettings] = React.useState<Settings>(DEFAULT_SETTINGS)
  const [commandOpen, setCommandOpen] = React.useState(false)

  // Whether the account's own copy has landed, so a change made in the gap
  // between mount and that response doesn't PUT a still-default object over
  // whatever was already saved. A ref, not state — read from inside a
  // `setSettings` updater, where a stale closure over React state would win.
  const loaded = React.useRef(false)
  const writeTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const pending = React.useRef<Settings | null>(null)

  const flush = React.useCallback(() => {
    if (writeTimer.current !== null) {
      clearTimeout(writeTimer.current)
      writeTimer.current = null
    }
    if (pending.current) {
      persist(pending.current)
      pending.current = null
    }
  }, [])

  const schedule = React.useCallback(
    (next: Settings) => {
      pending.current = next
      if (writeTimer.current !== null) clearTimeout(writeTimer.current)
      writeTimer.current = setTimeout(flush, WRITE_DEBOUNCE_MS)
    },
    [flush]
  )

  // Settings belong to the account, not the browser: load them on login. This
  // provider lives in the root layout (mounted on /login and /signup too), so
  // the fetch is skipped whenever there's no session — same reasoning as
  // `ChatsProvider`. Logged-out is *derived* below rather than written here,
  // for the same reason `ChatsProvider` derives its empty list instead of
  // clearing state in this effect: it keeps the previous session's values out
  // of `settings` without a synchronous `setState` in an effect body.
  React.useEffect(() => {
    loaded.current = false
    if (!isAuthenticated) {
      // Discard rather than flush: the token is already gone, so a write here
      // would only 401 and bounce back through `handleSessionExpired`.
      if (writeTimer.current !== null) clearTimeout(writeTimer.current)
      writeTimer.current = null
      pending.current = null
      return
    }
    let active = true
    apiFetch("/user/me/settings")
      .then(async (response) => {
        if (!response.ok) throw new Error("Unable to load settings.")
        return (await response.json()) as { settings: Partial<Settings> }
      })
      .then((data) => {
        if (!active) return
        // Merged over defaults so a setting added after this account last
        // saved picks up its default rather than coming back `undefined`.
        setSettings({ ...DEFAULT_SETTINGS, ...data.settings })
      })
      .catch(() => {
        // Stay on defaults; the next change still saves normally.
      })
      .finally(() => {
        if (active) loaded.current = true
      })
    return () => {
      active = false
    }
  }, [isAuthenticated])

  // A pending debounced write must not be lost to a closed tab or a logout.
  React.useEffect(() => {
    return () => flush()
  }, [flush])

  const setSetting = React.useCallback(
    <K extends keyof Settings>(key: K, value: Settings[K]) => {
      setSettings((prev) => {
        const next = { ...prev, [key]: value }
        if (loaded.current) schedule(next)
        return next
      })
    },
    [schedule]
  )

  const resetSettings = React.useCallback(() => {
    setSettings(DEFAULT_SETTINGS)
    if (loaded.current) schedule(DEFAULT_SETTINGS)
  }, [schedule])

  const value = React.useMemo(
    () => ({
      // Derived, not stored: logging out must not leave the previous
      // session's values on screen until the next fetch happens to replace
      // them (the same reasoning `ChatsProvider` applies to its list).
      settings: isAuthenticated ? settings : DEFAULT_SETTINGS,
      setSetting,
      resetSettings,
      commandOpen,
      setCommandOpen,
    }),
    [isAuthenticated, settings, setSetting, resetSettings, commandOpen]
  )

  return (
    <SettingsContext.Provider value={value}>
      {children}
    </SettingsContext.Provider>
  )
}

export function useSettings(): SettingsContextValue {
  const ctx = React.useContext(SettingsContext)
  if (!ctx)
    throw new Error("useSettings must be used within a SettingsProvider")
  return ctx
}
