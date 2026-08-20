"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"
import { attachmentPath } from "@/lib/attachments"

/**
 * How much of a text attachment is put in the DOM.
 *
 * The files people attach are not sized for a `<pre>`: a novel runs to five
 * megabytes, and laying that out as one text node stalls the tab for seconds.
 * Everything past this is still stored, still downloadable, and simply not
 * rendered — which the viewer says out loud rather than leaving the reader to
 * wonder where the rest went.
 */
export const PREVIEW_CHARS = 200_000

export interface AttachmentText {
  text: string | null
  /** Characters withheld, or 0 when the whole file is on screen. */
  withheld: number
  failed: boolean
}

/** The decoded contents of a text attachment, capped at `PREVIEW_CHARS`. */
export function useAttachmentText(attachmentId: string): AttachmentText {
  const [text, setText] = React.useState<string | null>(null)
  const [withheld, setWithheld] = React.useState(0)
  const [failed, setFailed] = React.useState(false)

  // Reset during render, not in the effect: pointing this at another file must
  // not leave a frame showing the previous one's contents.
  const [loadedFor, setLoadedFor] = React.useState(attachmentId)
  if (loadedFor !== attachmentId) {
    setLoadedFor(attachmentId)
    setText(null)
    setWithheld(0)
    setFailed(false)
  }

  React.useEffect(() => {
    let active = true

    apiFetch(attachmentPath(attachmentId))
      .then(async (response) => {
        if (!response.ok) throw new Error("unavailable")
        return response.text()
      })
      .then((whole) => {
        if (!active) return
        setText(whole.slice(0, PREVIEW_CHARS))
        setWithheld(Math.max(0, whole.length - PREVIEW_CHARS))
      })
      .catch(() => {
        if (active) setFailed(true)
      })

    return () => {
      active = false
    }
  }, [attachmentId])

  return { text, withheld, failed }
}
