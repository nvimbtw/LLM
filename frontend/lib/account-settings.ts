import { apiFetch, getToken } from "./api"
import { DEFAULT_SETTINGS, type Settings } from "./settings"

/** Returns whether unknown JSON can be read as a settings object, excluding null and arrays. */
function settingsObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

/** Reads a finite numeric preference, retaining the default when older or malformed blobs omit a usable value. */
function numericSetting(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback
}

/** Validates an account settings blob and merges known, correctly typed values over current defaults; invalid containers throw. */
export function parseAccountSettings(value: unknown): Settings {
  if (!settingsObject(value)) throw new Error("Unable to load settings.")
  return {
    useLiveThrobber:
      typeof value.useLiveThrobber === "boolean"
        ? value.useLiveThrobber
        : DEFAULT_SETTINGS.useLiveThrobber,
    showReasoning:
      typeof value.showReasoning === "boolean"
        ? value.showReasoning
        : DEFAULT_SETTINGS.showReasoning,
    notifyOnFinish:
      typeof value.notifyOnFinish === "boolean"
        ? value.notifyOnFinish
        : DEFAULT_SETTINGS.notifyOnFinish,
    systemPrompt:
      typeof value.systemPrompt === "string"
        ? value.systemPrompt
        : DEFAULT_SETTINGS.systemPrompt,
    temperature: numericSetting(
      value.temperature,
      DEFAULT_SETTINGS.temperature
    ),
    topK: numericSetting(value.topK, DEFAULT_SETTINGS.topK),
    topP: numericSetting(value.topP, DEFAULT_SETTINGS.topP),
    contextMultiplier: numericSetting(
      value.contextMultiplier,
      DEFAULT_SETTINGS.contextMultiplier
    ),
  }
}

/** Fetches and validates this session's account settings; HTTP or malformed response failures reject without authorizing writes. */
export async function loadAccountSettings(token: string): Promise<Settings> {
  const response = await apiFetch("/user/me/settings", {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!response.ok) throw new Error("Unable to load settings.")
  const data: unknown = await response.json()
  if (!settingsObject(data)) throw new Error("Unable to load settings.")
  return parseAccountSettings(data.settings)
}

/**
 * Writes preferences only while their owning token is still current, keeping stale timers away from replacement accounts.
 * Failed best-effort writes preserve local state; a later change retries the current complete value.
 */
export function persistAccountSettings(
  token: string,
  settings: Settings
): void {
  if (getToken() !== token) return
  void apiFetch("/user/me/settings", {
    method: "PUT",
    headers: { Authorization: `Bearer ${token}` },
    body: JSON.stringify({ settings }),
  }).catch(() => {})
}
