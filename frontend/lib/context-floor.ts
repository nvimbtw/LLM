import { apiFetch } from "@/lib/api"

/** What `/chats/{id}/context-floor` reports: where `/prune` would cut this
 *  chat's history right now, under the model's own (un-multiplied)
 *  `ctx_window` — see the LLM engine's `context_floor` for what decides it. */
export interface ContextFloor {
  /** The message a future turn should stop reaching past. Null when the
   *  chat has no messages yet. */
  floorMessageId: string | null
  prunedTurns: number
  prunedTokens: number
}

/** Read-only: nothing is written by this call, on either side of it. Storing
 *  the answer — so a later turn actually excludes what it reports — is
 *  `setContextFloor`'s job, not this one's. */
export async function fetchContextFloor(
  chatId: string,
  modelName: string
): Promise<ContextFloor> {
  const response = await apiFetch(
    `/chats/${chatId}/context-floor?model_name=${encodeURIComponent(modelName)}`
  )
  if (!response.ok) {
    throw new Error("Could not compute a context floor for this chat.")
  }
  const body = await response.json()
  return {
    floorMessageId: body.floor_message_id,
    prunedTurns: body.pruned_turns,
    prunedTokens: body.pruned_tokens,
  }
}

/** What's remembered between turns once `/prune` has run once. Just enough to
 *  resend the floor (`messageId`) and to shrink the composer's own token
 *  count by the right amount (`prunedTokens`) without a second round trip.
 *
 *  `modelId` is the model the cut was computed against — a floor sized to one
 *  model's `ctx_window` doesn't mean anything for another, so `use-chat.ts`
 *  only trusts a stored floor while it still matches the model currently
 *  selected, and otherwise treats the chat as unpruned until `/prune` runs
 *  again under the model that's actually answering now. */
export interface StoredFloor {
  messageId: string
  prunedTokens: number
  modelId: string
}

const STORAGE_PREFIX = "context-floor:"

/** Browser-local, per chat, the same tier as every other chat setting in
 *  `lib/settings.ts` — not a durable record. A lost floor (cleared site data,
 *  a different browser or device) just means the next turn falls back to the
 *  full history again, not a broken one. If that ever needs to survive those,
 *  this is the point to reconsider: persist the floor against the chat on the
 *  engine side instead of trusting the client to keep resending it — see
 *  `docs/frontend/README.md`'s note on `/prune`. */
export function getContextFloor(chatId: string): StoredFloor | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_PREFIX + chatId)
    return raw ? (JSON.parse(raw) as StoredFloor) : null
  } catch {
    return null
  }
}

export function setContextFloor(chatId: string, floor: StoredFloor): void {
  try {
    window.localStorage.setItem(STORAGE_PREFIX + chatId, JSON.stringify(floor))
  } catch {
    // Best-effort. A floor that fails to save just means the next turn falls
    // back to the full history, same as one that was never set.
  }
}
