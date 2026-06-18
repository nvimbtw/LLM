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

// Deterministic mock series so the page renders something believable until
// the backend exposes a usage endpoint.
function buildSeries(days: number): UsageDay[] {
  const out: UsageDay[] = []
  const end = new Date("2026-06-17T00:00:00Z")

  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(end)
    d.setUTCDate(end.getUTCDate() - i)

    const weekend = d.getUTCDay() === 0 || d.getUTCDay() === 6
    const base = 48 + Math.round(26 * Math.sin(i / 3.2))
    const requests = Math.max(6, base + (i % 5) * 5 - (weekend ? 22 : 0))
    const input = requests * (240 + (i % 9) * 14)
    const output = requests * (520 + (i % 6) * 24)

    out.push({ date: d.toISOString().slice(0, 10), input, output, requests })
  }

  return out
}

const SERIES = buildSeries(30)

const TOTAL_TOKENS = SERIES.reduce((sum, d) => sum + d.input + d.output, 0)
const TOTAL_REQUESTS = SERIES.reduce((sum, d) => sum + d.requests, 0)

const BY_MODEL: ModelUsage[] = [
  { id: "claude-opus-4-8", name: "Opus 4.8", tokens: Math.round(TOTAL_TOKENS * 0.46), requests: Math.round(TOTAL_REQUESTS * 0.34) },
  { id: "claude-sonnet-4-6", name: "Sonnet 4.6", tokens: Math.round(TOTAL_TOKENS * 0.39), requests: Math.round(TOTAL_REQUESTS * 0.44) },
  { id: "claude-haiku-4-5", name: "Haiku 4.5", tokens: Math.round(TOTAL_TOKENS * 0.15), requests: Math.round(TOTAL_REQUESTS * 0.22) },
]

const MOCK_USAGE: Usage = {
  totalRequests: TOTAL_REQUESTS,
  totalTokens: TOTAL_TOKENS,
  totalCost: Math.round((TOTAL_TOKENS / 1_000_000) * 9.4 * 100) / 100,
  series: SERIES,
  byModel: BY_MODEL,
}

export function useUsage() {
  // TODO: replace with real fetch to backend once API is up
  return {
    usage: MOCK_USAGE,
    isLoading: false,
    error: null as Error | null,
  }
}
