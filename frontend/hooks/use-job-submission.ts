"use client"

import * as React from "react"
import { useSessionToken } from "./use-auth"
import { JobSubmission } from "@/lib/job-submission"

/** Supplies an empty subscription before a captured-session queue owner has mounted. */
const noSubscription = () => () => {}
/** Supplies no observed state while the browser session is unavailable or replacing its owner. */
const noSnapshot = () => null

/** Owns explicit queue submission for one actual session/resource; replacing either detaches transport without cancelling durable execution. */
export function useJobSubmission(resource: string) {
  const token = useSessionToken()
  const [holder, setHolder] = React.useState<{
    token: string
    resource: string
    owner: JobSubmission
  } | null>(null)
  React.useEffect(() => {
    if (!token) return
    const owner = new JobSubmission(token)
    let active = true
    void Promise.resolve().then(() => {
      if (active) setHolder({ token, resource, owner })
    })
    return () => {
      active = false
      owner.dispose()
    }
  }, [token, resource])
  const owner =
    holder?.token === token && holder.resource === resource
      ? holder.owner
      : null
  const state = React.useSyncExternalStore(
    owner?.subscribe ?? noSubscription,
    owner?.snapshot ?? noSnapshot,
    noSnapshot
  )
  return { owner, state, pending: !owner || Boolean(state?.pending) }
}
