export type ModelTier = "flagship" | "balanced" | "fast"

export interface Model {
  id: string
  name: string
  family: string
  tier: ModelTier
  /** Parameter count in billions. 0 means undisclosed. */
  params: number
  /** Context window in tokens. */
  contextWindow: number
  description: string
}

// Illustrative mock data — replace once the backend exposes a models endpoint.
const MOCK_MODELS: Model[] = [
  {
    id: "claude-opus-4-8",
    name: "Opus 4.8",
    family: "Claude",
    tier: "flagship",
    params: 405,
    contextWindow: 200_000,
    description:
      "The most capable model, built for deep reasoning and complex, multi-step work.",
  },
  {
    id: "claude-sonnet-4-6",
    name: "Sonnet 4.6",
    family: "Claude",
    tier: "balanced",
    params: 70,
    contextWindow: 200_000,
    description:
      "A balance of speed and intelligence for everyday tasks at scale.",
  },
  {
    id: "claude-haiku-4-5",
    name: "Haiku 4.5",
    family: "Claude",
    tier: "fast",
    params: 8,
    contextWindow: 200_000,
    description:
      "The fastest, most compact model for low-latency, high-volume use.",
  },
]

export function useModels() {
  // TODO: replace with real fetch to backend once API is up
  return {
    models: MOCK_MODELS,
    isLoading: false,
    error: null as Error | null,
  }
}
