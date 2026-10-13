"use client"

import * as React from "react"
import { ComposerAttachments } from "@/lib/composer-attachments"

/** Owns one composer's upload draft and releases its local preview URLs when it unmounts. */
export function useComposerAttachments() {
  const [draft] = React.useState(() => new ComposerAttachments())
  const attachments = React.useSyncExternalStore(
    draft.subscribe,
    draft.getSnapshot,
    draft.getSnapshot
  )
  React.useEffect(() => () => draft.dispose(), [draft])
  return {
    attachments,
    addFiles: draft.addFiles,
    removeAttachment: draft.remove,
    releaseAttachments: draft.release,
    discardAttachments: draft.discard,
  }
}
