"use client"

import * as React from "react"

import { apiFetch } from "@/lib/api"

export interface UsageDay {
  date: string // ISO date (YYYY-MM-DD)
  input: number // tokens asked
  output: number // tokens generated
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

// These are local models with no per-token billing, so cost is notional: a flat
// rate per million tokens, purely so the page has a figure people recognize.
const USD_PER_MILLION_TOKENS = 9.4

interface UsagePayload {
  series: { date: string; asked: number; generated: number; replies: number }[]
  by_model: {
    model_id: string | null
    name: string | null
    tokens: number
    replies: number
  }[]
}

function mapUsage(payload: UsagePayload): Usage {
  const series = payload.series.map((day) => ({
    date: day.date,
    input: day.asked,
    output: day.generated,
    requests: day.replies,
  }))
  const totalTokens = series.reduce((sum, d) => sum + d.input + d.output, 0)
  return {
    totalRequests: series.reduce((sum, d) => sum + d.requests, 0),
    totalTokens,
    totalCost: (totalTokens / 1_000_000) * USD_PER_MILLION_TOKENS,
    series,
    byModel: payload.by_model.map((m) => ({
      id: m.model_id ?? "",
      // A model since deleted still owns the tokens it generated.
      name: m.name ?? "Deleted model",
      tokens: m.tokens,
      requests: m.replies,
    })),
  }
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
        return mapUsage((await res.json()) as UsagePayload)
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
