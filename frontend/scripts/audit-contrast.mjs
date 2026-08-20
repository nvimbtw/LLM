// Contrast audit of every token pair in `app/globals.css`, both modes.
//
//   node scripts/audit-contrast.mjs
//
// Floors: 4.5 for text (AA body — the app sets almost no text at "large" sizes,
// and its eyebrows are small and letterspaced), 3.0 for graphical objects that
// carry meaning (chart marks, form-control boundaries), and 1.8 for hairlines,
// which carry none and are held only to "still visible on a bright screen".
// One row fails by design: `--tier-low`, whose reasoning is on the token itself.

import { themeTokens } from "./theme-tokens.mjs"

const { light, dark } = themeTokens()

const srgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255)
const lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)
const L = (h) => { const [r, g, b] = srgb(h).map(lin); return 0.2126 * r + 0.7152 * g + 0.0722 * b }
const ratio = (a, b) => { const [x, y] = [L(a), L(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05) }
// mix fg over bg at `pct` alpha, like color-mix in a tint
const mix = (fg, bg, pct) => "#" + [0,1,2].map(i => {
  const f = parseInt(fg.slice(1+i*2, 3+i*2), 16), b = parseInt(bg.slice(1+i*2, 3+i*2), 16)
  return Math.round(f*pct + b*(1-pct)).toString(16).padStart(2,"0")
}).join("")

// [label, fg, bg, floor]  — 4.5 body text, 3.0 large text / UI graphics
const pairs = (t) => [
  ["foreground / background",            t.foreground,           t.background, 4.5],
  ["foreground / card",                  t["card-foreground"],   t.card,       4.5],
  ["foreground / popover",               t["popover-foreground"],t.popover,    4.5],
  ["foreground / secondary",             t["secondary-foreground"], t.secondary, 4.5],
  ["foreground / accent",                t["accent-foreground"], t.accent,     4.5],
  ["foreground / muted surface",         t.foreground,           t.muted,      4.5],
  ["muted-foreground / background",      t["muted-foreground"],  t.background, 4.5],
  ["muted-foreground / card",            t["muted-foreground"],  t.card,       4.5],
  ["muted-foreground / muted",           t["muted-foreground"],  t.muted,      4.5],
  ["muted-foreground / secondary",       t["muted-foreground"],  t.secondary,  4.5],
  ["primary / background",               t.primary,              t.background, 4.5],
  ["primary / card",                     t.primary,              t.card,       4.5],
  ["primary-foreground / primary",       t["primary-foreground"],t.primary,    4.5],
  ["destructive / background",           t.destructive,          t.background, 4.5],
  ["destructive / card",                 t.destructive,          t.card,       4.5],
  // Selected text sets its own foreground, so this pair is the whole story —
  // whatever was selected, this is what it reads as while it is.
  ["selection-fg / selection",           t["selection-foreground"], t.selection, 4.5],
  ["border / background",                t.border,               t.background, 1.8],
  ["border / card",                      t.border,               t.card,       1.8],
  ["input / background",                 t.input,                t.background, 3.0],
  ["ring / background",                  t.ring,                 t.background, 3.0],
  ["sidebar-fg / sidebar",               t["sidebar-foreground"],t.sidebar,    4.5],
  ["sidebar-border / sidebar",           t["sidebar-border"],    t.sidebar,    1.8],
  ["sidebar-accent-fg / sidebar-accent", t["sidebar-accent-foreground"], t["sidebar-accent"], 4.5],
  ["sidebar-primary / sidebar",          t["sidebar-primary"],   t.sidebar,    3.0],
  ["throbber-base / background",         t["throbber-base"],     t.background, 3.0],
  ["throbber-spark / background",        t["throbber-spark"],    t.background, 3.0],
  ...[1,2,3,4,5].map(n => [`chart-${n} / background`, t[`chart-${n}`], t.background, 3.0]),
  ["chart-approved / background",        t["chart-approved"],    t.background, 3.0],
  ["chart-rejected / background",        t["chart-rejected"],    t.background, 3.0],
  ...["train","benchmark","chat","tokenize"].map(k =>
      [`kind-${k} on its own tint`, t[`kind-${k}`], mix(t[`kind-${k}`], t.card, 0.16), 4.5]),
  ...["low","med","high"].map(k => [`tier-${k} / background`, t[`tier-${k}`], t.background, 3.0]),
]

for (const [name, tokens] of [["LIGHT (warm paper)", light], ["DARK (dragon)", dark]]) {
  console.log(`\n──── ${name} ${"─".repeat(46 - name.length)}`)
  const rows = pairs(tokens)
    .filter(([, fg, bg]) => fg && bg)
    .map(([label, fg, bg, floor]) => ({ label, fg, bg, floor, r: ratio(fg, bg) }))
  for (const row of rows.sort((a, b) => a.r / a.floor - b.r / b.floor)) {
    const pass = row.r >= row.floor
    const mark = pass ? "  ok  " : " FAIL "
    console.log(`${mark} ${row.r.toFixed(2).padStart(5)} / ${row.floor.toFixed(1)}  ${row.label.padEnd(34)} ${row.fg} on ${row.bg}`)
  }
}
