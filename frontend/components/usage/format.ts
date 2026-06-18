export function formatCompact(n: number) {
  if (n >= 1_000_000_000) return `${trim(n / 1_000_000_000)}B`
  if (n >= 1_000_000) return `${trim(n / 1_000_000)}M`
  if (n >= 1_000) return `${trim(n / 1_000)}K`
  return `${n}`
}

export function formatNumber(n: number) {
  return n.toLocaleString("en-US")
}

export function formatCurrency(n: number) {
  return n.toLocaleString("en-US", { style: "currency", currency: "USD" })
}

function trim(n: number) {
  return n % 1 ? n.toFixed(1) : `${n}`
}
