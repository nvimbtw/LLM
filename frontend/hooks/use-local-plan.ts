"use client"

import * as React from "react"
import { parsePlan, type Plan } from "@/lib/plan"
import { PLAN_CHANGED, readPlan, writePlan } from "@/lib/plan-storage"

function subscribe(listener: () => void) {
  window.addEventListener(PLAN_CHANGED, listener)
  window.addEventListener("storage", listener)
  return () => {
    window.removeEventListener(PLAN_CHANGED, listener)
    window.removeEventListener("storage", listener)
  }
}
const serverSnapshot = () => null

export function useLocalPlan(chatId?: string) {
  const [local, setLocal] = React.useState<Plan | null>(null)
  const snapshot = React.useCallback(
    () => (chatId ? readPlan(chatId) : null),
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
