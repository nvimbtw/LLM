"use client"

import * as React from "react"

const checkedNavigations = new WeakSet<MouseEvent>()

/** Warns before link navigation or browser unload only when dirty edits could not be retained in this tab; healthy drafts navigate without prompts. */
export function useDraftRetentionWarning(atRisk: boolean) {
  React.useEffect(() => {
    if (!atRisk) return
    /** Requests the browser's native loss warning when the tab cannot recover unsaved fields. */
    const beforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault()
      event.returnValue = ""
    }
    /** Leaves failed-to-retain fields in place when the user declines a link's explicit loss confirmation. */
    const beforeNavigate = (event: MouseEvent) => {
      if (
        event.defaultPrevented ||
        checkedNavigations.has(event) ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey ||
        event.altKey ||
        event.button !== 0
      )
        return
      if (
        !(event.target instanceof Element) ||
        !event.target.closest("a[href]")
      )
        return
      const anchor = event.target.closest<HTMLAnchorElement>("a[href]")!
      if (anchor.target === "_blank" || anchor.hasAttribute("download")) return
      const destination = new URL(anchor.href)
      if (
        destination.origin === location.origin &&
        destination.pathname === location.pathname &&
        destination.search === location.search
      )
        return
      checkedNavigations.add(event)
      if (
        !window.confirm(
          "This draft could not be retained. Leave and discard unsaved changes?"
        )
      ) {
        event.preventDefault()
        event.stopPropagation()
      }
    }
    window.addEventListener("beforeunload", beforeUnload)
    document.addEventListener("click", beforeNavigate, true)
    return () => {
      window.removeEventListener("beforeunload", beforeUnload)
      document.removeEventListener("click", beforeNavigate, true)
    }
  }, [atRisk])
}
