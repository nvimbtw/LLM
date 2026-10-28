import { getToken } from "./api"
import { identityDigest } from "./architecture/decoder"

const MAX_DRAFT_CHARACTERS = 600000

export interface PageDraftIdentity {
  token: string
  resource: string
}

/** Names one tab-local resource draft without placing bearer credentials in browser storage keys. */
function draftKey(identity: PageDraftIdentity): string {
  return `slate-page-draft:${identityDigest(identity.token)}:${identity.resource}`
}

/** Validates a known set of text fields in local JSON, retaining empty and multiline strings for ordinary form validation. */
export function readDraftStrings<K extends string>(
  keys: readonly K[],
  value: unknown
): Record<K, string> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("The saved page draft is invalid.")
  const record = value as Record<string, unknown>
  const fields = {} as Record<K, string>
  for (const key of keys) {
    if (typeof record[key] !== "string")
      throw new Error("The saved page draft contains an invalid text field.")
    fields[key] = record[key]
  }
  return fields
}

/** Restores validated fields for the same session/resource, retaining a visible warning if the server baseline changed or local storage failed. */
export function readPageDraft<T>(
  identity: PageDraftIdentity | null,
  saved: T,
  parse: (value: unknown) => T
): { fields: T; error: string | null; retained: boolean } {
  if (!identity) return { fields: saved, error: null, retained: true }
  try {
    const raw = sessionStorage.getItem(draftKey(identity))
    if (!raw) return { fields: saved, error: null, retained: true }
    if (raw.length > MAX_DRAFT_CHARACTERS)
      throw new Error("The saved page draft is too large.")
    const value: unknown = JSON.parse(raw)
    if (
      !value ||
      typeof value !== "object" ||
      !("version" in value) ||
      value.version !== 1 ||
      !("fields" in value) ||
      !("baseline" in value) ||
      typeof value.baseline !== "string"
    )
      throw new Error("The saved page draft is invalid.")
    return {
      fields: parse(value.fields),
      retained: true,
      error:
        value.baseline === JSON.stringify(saved)
          ? null
          : "Saved values changed while you were away. Review your restored edits before saving.",
    }
  } catch (cause) {
    return {
      fields: saved,
      retained: false,
      error:
        cause instanceof Error
          ? cause.message
          : "The page draft could not be restored.",
    }
  }
}

/** Saves an incomplete resource draft in this tab only; clean fields remove its entry. Stale sessions and storage failures reject visibly. */
export function writePageDraft<T>(
  identity: PageDraftIdentity | null,
  saved: T,
  fields: T
): void {
  if (!identity || getToken() !== identity.token)
    throw new Error("Your session changed. This page draft was not retained.")
  const baseline = JSON.stringify(saved)
  if (JSON.stringify(fields) === baseline) {
    sessionStorage.removeItem(draftKey(identity))
    return
  }
  const raw = JSON.stringify({ version: 1, baseline, fields })
  if (raw.length > MAX_DRAFT_CHARACTERS)
    throw new Error("This page draft is too large to retain while browsing.")
  sessionStorage.setItem(draftKey(identity), raw)
}
