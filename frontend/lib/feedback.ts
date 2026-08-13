import { apiFetch } from "./api"

/** A user's verdict on one assistant reply. `null` is "no opinion". */
export type Feedback = "up" | "down"

/**
 * Sets or clears the thumbs on one reply.
 *
 * PUT rather than POST, and the desired state rather than a toggle: a retry or a
 * double-click then can't leave the rating flipped away from what the button shows.
 */
export async function setMessageFeedback(
  messageId: string,
  feedback: Feedback | null
): Promise<void> {
  const response = await apiFetch(`/messages/${messageId}/feedback`, {
    method: "PUT",
    body: JSON.stringify({ feedback }),
  })
  if (!response.ok) throw new Error("Unable to save your feedback.")
}
