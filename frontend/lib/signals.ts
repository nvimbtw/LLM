import { apiFetch } from "./api"

/**
 * Preference signal: the thumbs a KTO run trains on.
 *
 * Two sources feed one pool — replies rated in a conversation, and replies a
 * gather job generated purely to be rated. The backend counts and exports them
 * together, so nothing here distinguishes them either.
 */

/** A verdict on one reply. Matches the backend `message_feedback` enum. */
export type Verdict = "up" | "down"

/**
 * How much signal exists, and whether it is enough to tune on.
 *
 * **Two stages, and they are not the same number.** The *seed* is replies a person
 * rated by hand — a few dozen, enough to calibrate the auto-scoring judge against
 * actual taste. The *catalogue* is what a tune trains on: the seed plus everything
 * auto-labelling produced from it, two orders of magnitude larger. Treating the
 * seed bar as the training bar is what allows a tune on thirty examples, which
 * memorises them instead of learning from them.
 *
 * `ready` (both bars cleared) is computed server-side rather than by comparing
 * counts here: the same functions back the enqueue check, so the button and the API
 * cannot end up disagreeing about whether a run can start.
 */
export interface SignalStatus {
  // ── seed: what you rated ──
  desirable: number
  undesirable: number
  total: number
  /** Generated and waiting for a verdict — "finish rating" rather than "gather". */
  pending: number
  required: number
  required_per_class: number
  /** Enough hand-rated replies to calibrate the judge, i.e. auto-labelling can run. */
  seeded: boolean
  seed_reason: string | null

  // ── catalogue: what a tune trains on ──
  catalogue: number
  catalogue_approved: number
  catalogue_rejected: number
  /** How much of the catalogue came from auto-labelling rather than your thumbs. */
  catalogue_auto: number
  catalogue_required: number
  /** Epochs this catalogue size allows; a run asking for more is clamped to it. */
  max_epochs: number

  ready: boolean
  /** Why not, when it isn't. Null when ready. */
  blocked_reason: string | null
}

/** One generated reply awaiting a verdict. */
export interface PendingSample {
  id_sample: string
  id_job: string | null
  prompt: string
  response: string
}

export async function getSignalStatus(): Promise<SignalStatus> {
  const res = await apiFetch("/signals")
  if (!res.ok) throw new Error("Failed to load the signal count.")
  return res.json()
}

/**
 * Unrated samples, oldest first. `idJob` narrows to one batch; omitted, the whole
 * backlog comes back — so a batch left half-rated can be finished from anywhere.
 */
export async function listPendingSamples(idJob?: string): Promise<PendingSample[]> {
  const query = idJob ? `?id_job=${encodeURIComponent(idJob)}` : ""
  const res = await apiFetch(`/signals/pending${query}`)
  if (!res.ok) throw new Error("Failed to load samples to rate.")
  return res.json()
}

/** Records a verdict. `null` clears it and returns the sample to the queue. */
export async function rateSample(id: string, verdict: Verdict | null): Promise<void> {
  const res = await apiFetch(`/signals/${id}`, {
    method: "PUT",
    body: JSON.stringify({ feedback: verdict }),
  })
  if (!res.ok) throw new Error((await res.text()) || "Failed to save the rating.")
}

/**
 * Drops a sample. A reply nobody can form a verdict on carries no signal in
 * either direction, so it is deleted rather than left pending to be offered again.
 */
export async function skipSample(id: string): Promise<void> {
  const res = await apiFetch(`/signals/${id}`, { method: "DELETE" })
  if (!res.ok) throw new Error((await res.text()) || "Failed to skip the sample.")
}
