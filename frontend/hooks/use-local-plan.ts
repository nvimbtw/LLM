"use client"

import * as React from "react"
import { parsePlan, type Plan } from "@/lib/plan"
import { PLAN_CHANGED, readPlan, writePlan } from "@/lib/plan-storage"

// Last raw value read per chat. `useSyncExternalStore` calls its snapshot on
// every render of every subscriber — and the thread renders once per streamed
// frame — so reading local storage through this cache keeps a synchronous
// browser read off the render path. Any plan change clears it before the
// subscribers are told, which is what keeps the cache honest.
const cache = new Map<string, string | null>()

/** Returns the stored plan source for a chat, reading local storage at most once per change. */
function read(chatId: string): string | null {
  const cached = cache.get(chatId)
  if (cached !== undefined) return cached
  const value = readPlan(chatId)
  cache.set(chatId, value)
  return value
}

function subscribe(listener: () => void) {
  const invalidate = () => cache.clear()
  window.addEventListener(PLAN_CHANGED, invalidate)
  window.addEventListener("storage", invalidate)
  window.addEventListener(PLAN_CHANGED, listener)
  window.addEventListener("storage", listener)
  return () => {
    window.removeEventListener(PLAN_CHANGED, invalidate)
    window.removeEventListener("storage", invalidate)
    window.removeEventListener(PLAN_CHANGED, listener)
    window.removeEventListener("storage", listener)
  }
}
const serverSnapshot = () => null

export function useLocalPlan(chatId?: string) {
  const [local, setLocal] = React.useState<Plan | null>(null)
  const snapshot = React.useCallback(
    () => (chatId ? read(chatId) : null),
    [chatId]
  )
  const raw = React.useSyncExternalStore(subscribe, snapshot, serverSnapshot)
  const stored = React.useMemo(() => parsePlan(raw), [raw])
  const save = React.useCallback(
    (plan: Plan) => {
      if (chatId) writePlan(chatId, plan)
      else setLocal(plan)
    },
    [chatId]
  )
  return { plan: chatId ? stored : local, save }
}
