const INITIAL_SEED = 2166136261
const HASH_PRIME = 16777619

export interface ArtSnapshot {
  seed: number
  structure: number
  count: number
  source: "waiting" | "token ids" | "text fallback" | "mixed"
}

/** Mutable, constant-size stream state; consumed by the canvas without React renders. */
export class ArtStream {
  snapshot: ArtSnapshot
  private listeners = new Set<() => void>()

  constructor(identity = "slate") {
    this.snapshot = this.initial(identity)
  }

  private initial(identity: string): ArtSnapshot {
    const seed = hashText(INITIAL_SEED, identity)
    return { seed, structure: seed, count: 0, source: "waiting" }
  }

  reset(identity: string) {
    this.snapshot = this.initial(identity)
    this.emit()
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  wake() {
    this.emit()
  }

  private emit() {
    for (const listener of this.listeners) listener()
  }

  push(text: string, tokenId?: number) {
    const previous = this.snapshot
    const hasId = validTokenId(tokenId)
    let seed = mix(previous.seed, hasId ? 1 : 0)
    if (hasId) {
      // Encode all four bytes, including token zero; token ids are never text.
      for (let byte = 0; byte < 4; byte++)
        seed = mix(seed, (tokenId >>> (byte * 8)) & 255)
    } else {
      seed = hashText(seed, text)
    }
    const source = hasId ? "token ids" : "text fallback"
    this.snapshot = {
      ...previous,
      seed,
      count: previous.count + 1,
      source:
        previous.source === "waiting" || previous.source === source
          ? source
          : "mixed",
    }
    this.emit()
  }
}

export function validTokenId(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isInteger(value) &&
    value >= 0 &&
    value <= 0xffffffff
  )
}

function mix(seed: number, value: number): number {
  return Math.imul(seed ^ value, HASH_PRIME) >>> 0
}

function hashText(seed: number, text: string): number {
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i)
    seed = mix(mix(seed, code & 255), code >>> 8)
  }
  return seed
}
