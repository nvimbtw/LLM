"use client"

import * as React from "react"

import { getToken } from "@/lib/api"
import {
  loadAccountSettings,
  persistAccountSettings,
} from "@/lib/account-settings"
import { createSettingsWriter } from "@/lib/settings-writer"
import { useSessionToken } from "@/hooks/use-auth"
import {
  DEFAULT_SETTINGS,
  type Settings,
  type SettingGroup,
} from "@/lib/settings"

export type SettingsSection = SettingGroup | "usage" | "profile"

interface SettingsContextValue {
  settings: Settings
  setSetting: <K extends keyof Settings>(key: K, value: Settings[K]) => void
  resetSettings: () => void
  commandOpen: boolean
  setCommandOpen: (open: boolean) => void
  section: SettingsSection
  setSection: (section: SettingsSection) => void
  openSettings: (section?: SettingsSection) => void
}

const SettingsContext = React.createContext<SettingsContextValue | null>(null)

interface AccountSettingsOwner {
  token: string
  active: boolean
  loaded: boolean
  settings: Settings
  writer: ReturnType<typeof createSettingsWriter>
}

export interface SettingsProviderProps {
  children: React.ReactNode
}

/**
 * Provides session-owned account preferences and settings-dialog state; unmatched/unloaded identities show defaults.
 * Only successful reads authorize writes; failed reads retain defaults and keep mutations disabled.
 * Replacement sessions discard old timers and ignore delayed responses.
 */
export function SettingsProvider({ children }: SettingsProviderProps) {
  const token = useSessionToken()
  const [account, setAccount] = React.useState<{
    token: string | null
    settings: Settings
  }>({ token: null, settings: DEFAULT_SETTINGS })
  const ownerRef = React.useRef<AccountSettingsOwner | null>(null)
  const [commandOpen, setCommandOpen] = React.useState(false)
  const [section, setSection] = React.useState<SettingsSection>("appearance")
  /** Opens the shared settings dialog at the requested section, defaulting to Appearance. */
  const openSettings = React.useCallback(
    (section: SettingsSection = "appearance") => {
      setSection(section)
      setCommandOpen(true)
    },
    []
  )

  React.useEffect(() => {
    if (!token) {
      ownerRef.current = null
      return
    }
    const owner: AccountSettingsOwner = {
      token,
      active: true,
      loaded: false,
      settings: DEFAULT_SETTINGS,
      writer: createSettingsWriter((next) =>
        persistAccountSettings(token, next)
      ),
    }
    ownerRef.current = owner
    loadAccountSettings(token)
      .then((settings) => {
        if (!owner.active || getToken() !== token) return
        owner.settings = settings
        owner.loaded = true
        setAccount({ token, settings })
      })
      .catch(() => {})
    return () => {
      owner.active = false
      if (getToken() === token) owner.writer.flush()
      else owner.writer.discard()
    }
  }, [token])

  /** Changes a loaded account preference and schedules its complete value; unknown or replaced accounts cannot write. */
  const setSetting = React.useCallback(
    <K extends keyof Settings>(key: K, value: Settings[K]) => {
      const owner = ownerRef.current
      if (!owner?.active || !owner.loaded || getToken() !== owner.token) return
      const settings = { ...owner.settings, [key]: value }
      owner.settings = settings
      setAccount({ token: owner.token, settings })
      owner.writer.schedule(settings)
    },
    []
  )

  /** Restores defaults for the loaded current account, leaving unloaded/replaced account storage untouched. */
  const resetSettings = React.useCallback(() => {
    const owner = ownerRef.current
    if (!owner?.active || !owner.loaded || getToken() !== owner.token) return
    owner.settings = DEFAULT_SETTINGS
    setAccount({ token: owner.token, settings: DEFAULT_SETTINGS })
    owner.writer.schedule(DEFAULT_SETTINGS)
  }, [])

  const value = React.useMemo(
    () => ({
      settings:
        token !== null && account.token === token
          ? account.settings
          : DEFAULT_SETTINGS,
      setSetting,
      resetSettings,
      commandOpen,
      setCommandOpen,
      section,
      setSection,
      openSettings,
    }),
    [
      token,
      account,
      setSetting,
      resetSettings,
      commandOpen,
      section,
      openSettings,
    ]
  )

  return (
    <SettingsContext.Provider value={value}>
      {children}
    </SettingsContext.Provider>
  )
}

/** Returns settings state/actions from the nearest provider, throwing when mounted outside one. */
export function useSettings(): SettingsContextValue {
  const ctx = React.useContext(SettingsContext)
  if (!ctx)
    throw new Error("useSettings must be used within a SettingsProvider")
  return ctx
}
