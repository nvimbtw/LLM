import { apiFetch, getToken } from "./api"
import type { ThinkBudget } from "./think-budget"

export interface ComposerChoice {
  model: string
  budget: ThinkBudget
}
export const DEFAULT_COMPOSER_CHOICE: ComposerChoice = {
  model: "",
  budget: "auto",
}

/** Validates persisted model and budget picks, using defaults only for omitted fields in older account blobs. */
export function parseComposerChoice(value: unknown): ComposerChoice {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error("Unable to load composer choice.")
  const raw = value as Record<string, unknown>
  const model = raw.model ?? ""
  const budget = raw.budget ?? "auto"
  if (
    typeof model !== "string" ||
    (budget !== "auto" &&
      budget !== "low" &&
      budget !== "med" &&
      budget !== "high")
  )
    throw new Error("Unable to load composer choice.")
  return { model, budget }
}

/** Loads the chosen account's model/budget with cancellation; malformed or failed responses reject. */
export async function loadComposerChoice(
  token: string,
  signal?: AbortSignal
): Promise<ComposerChoice> {
  const response = await apiFetch("/user/me/composer-choice", {
    headers: { Authorization: `Bearer ${token}` },
    signal,
  })
  if (!response.ok) throw new Error("Unable to load composer choice.")
  const data: unknown = await response.json()
  if (typeof data !== "object" || data === null || !("composer_choice" in data))
    throw new Error("Unable to load composer choice.")
  return parseComposerChoice(data.composer_choice)
}

/** Persists a ready account's explicit pick immediately; dropped writes keep the local pick and retry on the next change. */
export function persistComposerChoice(
  token: string,
  choice: ComposerChoice
): void {
  if (getToken() !== token) return
  void apiFetch("/user/me/composer-choice", {
    method: "PUT",
    headers: { Authorization: `Bearer ${token}` },
    body: JSON.stringify({ composer_choice: choice }),
  }).catch(() => {})
}
