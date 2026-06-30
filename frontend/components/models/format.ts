// Shared number formatting for model cards and the model detail page.

const UNITS: [number, string][] = [
  [1e12, "T"],
  [1e9, "B"],
  [1e6, "M"],
  [1e3, "K"],
]

/** Parameter count, e.g. 8_442_304 -> "8.4M". 0 reads as undisclosed. */
export function formatParams(count: number): string {
  if (!count) return "Undisclosed"
  return formatCount(count)
}

/** Context window in tokens, e.g. 64 -> "64", 8192 -> "8K". */
export function formatContext(tokens: number): string {
  if (tokens >= 1_000_000) {
    const m = tokens / 1_000_000
    return `${m % 1 ? m.toFixed(1) : m}M`
  }
  if (tokens >= 1000) return `${Math.round(tokens / 1000)}K`
  return `${tokens}`
}

/** Compact count with a unit suffix, e.g. 11_206_778 -> "11.2M". */
export function formatCount(n: number): string {
  for (const [size, suffix] of UNITS) {
    if (n >= size) {
      const v = n / size
      return `${v % 1 ? v.toFixed(1) : v}${suffix}`
    }
  }
  return `${n}`
}

/** Grouped integer, e.g. 11206778 -> "11,206,778". */
export function formatNumber(n: number): string {
  return n.toLocaleString("en-US")
}
