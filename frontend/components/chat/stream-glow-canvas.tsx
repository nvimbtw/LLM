"use client"

import * as React from "react"

const DURATION = 480
const MAX_ACTIVE = 10

function sampleGlow(): string {
  if (typeof window === "undefined") return "#4f4f58"
  const dark = document.documentElement.classList.contains("dark")
  return dark ? "#a6a69c" : "#4f4f58"
}

function cssToRgb(v: string): [number, number, number] {
  if (v.startsWith("#")) {
    let h = v.replace("#", "")
    if (h.length === 3) h = h.split("").map((c) => c + c).join("")
    return [Number.parseInt(h.slice(0, 2), 16), Number.parseInt(h.slice(2, 4), 16), Number.parseInt(h.slice(4, 6), 16)]
  }
  const el = document.createElement("div")
  el.style.color = v
  document.body.appendChild(el)
  const c = getComputedStyle(el).color
  el.remove()
  const n = c.match(/[\d.]+/g)?.map(Number) ?? [20, 149, 138]
  return [n[0], n[1], n[2]]
}

export function StreamGlowCanvas() {
  const ref = React.useRef<HTMLCanvasElement>(null)
  const reduced = React.useSyncExternalStore(
    (cb) => {
      const q = window.matchMedia("(prefers-reduced-motion: reduce)")
      q.addEventListener("change", cb)
      return () => q.removeEventListener("change", cb)
    },
    () => window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    () => false
  )

  React.useEffect(() => {
    if (reduced) return
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext("2d", { alpha: true })
    if (!ctx) return

    const active = new Map<Element, number>()
    let frame = 0
    let glowRgb: [number, number, number] | null = null

    const ro = new MutationObserver((mut) => {
      for (const m of mut) {
        m.addedNodes.forEach((n) => {
          if (!(n instanceof Element)) return
          if (n.classList.contains("word-glow")) active.set(n, performance.now())
          n.querySelectorAll?.(".word-glow").forEach((el) => active.set(el, performance.now()))
        })
      }
      document.querySelectorAll(".word-glow").forEach((el) => {
        if (!active.has(el)) active.set(el, performance.now() - Math.random() * 120)
      })
    })
    ro.observe(document.body, { childList: true, subtree: true })
    document.querySelectorAll(".word-glow").forEach((el) => active.set(el, performance.now()))

    let last = 0
    const tick = (now: number) => {
      if (document.hidden) { frame = requestAnimationFrame(tick); return }
      if (now - last < 42) { frame = requestAnimationFrame(tick); return }
      last = now
      if (!glowRgb) glowRgb = cssToRgb(sampleGlow())
      const dpr = 1
      if (canvas.width !== Math.round(window.innerWidth * dpr) || canvas.height !== Math.round(window.innerHeight * dpr)) {
        canvas.width = Math.round(window.innerWidth * dpr)
        canvas.height = Math.round(window.innerHeight * dpr)
        canvas.style.width = window.innerWidth + "px"
        canvas.style.height = window.innerHeight + "px"
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight)
      if (active.size === 0) { frame = requestAnimationFrame(tick); return }
      ctx.globalCompositeOperation = "lighter"
      let n = 0
      for (const [el, start] of Array.from(active.entries())) {
        if (n++ >= MAX_ACTIVE) break
        if (!document.contains(el)) { active.delete(el); continue }
        const t = (now - start) / DURATION
        if (t >= 1) { active.delete(el); continue }
        const rect = el.getBoundingClientRect()
        if (rect.width < 1 || rect.height < 1) continue
        const eased = Math.pow(1 - t, 0.65)
        const alpha = eased * 0.032
        const cx = rect.left + rect.width / 2
        const cy = rect.top + rect.height / 2
        const r = Math.max(rect.width, rect.height) * 0.45 + 4 * eased
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r)
        const [rr, gg, bb] = glowRgb
        g.addColorStop(0, `rgba(${rr},${gg},${bb},${alpha.toFixed(3)})`)
        g.addColorStop(1, `rgba(${rr},${gg},${bb},0)`)
        ctx.fillStyle = g
        ctx.beginPath()
        ctx.arc(cx, cy, r, 0, Math.PI * 2)
        ctx.fill()
      }
      ctx.globalCompositeOperation = "source-over"
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => { cancelAnimationFrame(frame); ro.disconnect() }
  }, [reduced])

  if (reduced) return null
  return <canvas ref={ref} aria-hidden className="pointer-events-none fixed inset-0 z-0 opacity-30 dark:opacity-25" style={{ mixBlendMode: "soft-light" }} />
}
