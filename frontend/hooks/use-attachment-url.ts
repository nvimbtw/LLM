"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"
import { attachmentPath } from "@/lib/attachments"

/**
 * A displayable URL for a stored attachment.
 *
 * The bytes are behind the same bearer token every other call carries, and an
 * `<img src>` cannot send a header — so the file is fetched and handed back as
 * an object URL instead. Revoked when the component goes or the id changes,
 * which is the whole reason this is a hook and not a function.
 *
 * Null while it loads and after a failure; `failed` is what tells the two
 * apart, since a broken image and one that has not arrived yet should not look
 * the same.
 */
export function useAttachmentUrl(attachmentId: string): {
  url: string | null
  failed: boolean
} {
  const [url, setUrl] = React.useState<string | null>(null)
  const [failed, setFailed] = React.useState(false)

  // Reset during render rather than in the effect below: pointing this hook at
  // a different file must not leave a frame showing the previous one, and a
  // reset written from the effect lands one render too late to prevent that.
  const [loadedFor, setLoadedFor] = React.useState(attachmentId)
  if (loadedFor !== attachmentId) {
    setLoadedFor(attachmentId)
    setUrl(null)
    setFailed(false)
  }

  React.useEffect(() => {
    let active = true
    let objectUrl: string | null = null

    apiFetch(attachmentPath(attachmentId))
      .then(async (response) => {
        if (!response.ok) throw new Error("unavailable")
        return response.blob()
      })
      .then((blob) => {
        // Revoke rather than show it: the effect was cleaned up while the
        // request was in flight, so nothing will ever release this one.
        objectUrl = URL.createObjectURL(blob)
        if (!active) {
          URL.revokeObjectURL(objectUrl)
          return
        }
        setUrl(objectUrl)
      })
      .catch(() => {
        if (active) setFailed(true)
      })

    return () => {
      active = false
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [attachmentId])

  return { url, failed }
}
