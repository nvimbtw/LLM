"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"

export interface UsageDay {
  date: string // ISO date (YYYY-MM-DD)
  input: number // input tokens
  output: number // output tokens
  requests: number
}

export interface ModelUsage {
  id: string
  name: string
  tokens: number
  requests: number
}

export interface Usage {
  totalRequests: number
  totalTokens: number
  totalCost: number // USD
  series: UsageDay[]
  byModel: ModelUsage[]
}

const EMPTY_USAGE: Usage = {
  totalRequests: 0,
  totalTokens: 0,
  totalCost: 0,
  series: [],
  byModel: [],
}

export function useUsage() {
  const [usage, setUsage] = React.useState<Usage>(EMPTY_USAGE)
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<Error | null>(null)

  React.useEffect(() => {
    let active = true
    apiFetch(`/usage`)
      .then(async (res) => {
        if (!res.ok) throw new Error("Failed to load usage.")
        return (await res.json()) as Usage
      })
      .then((data) => {
        if (active) {
          setUsage(data)
          setError(null)
        }
      })
      .catch((err) => {
        if (active)
          setError(
            err instanceof Error ? err : new Error("Failed to load usage.")
          )
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
    }
  }, [])

  return { usage, isLoading, error }
}
