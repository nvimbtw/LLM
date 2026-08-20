// The theme's perceptual audit, beyond WCAG.
//
//   node scripts/audit-theme.mjs
//
// WCAG contrast (audit-contrast.mjs) is the *floor*. This tool is what would
// have caught the reason the theme ached despite the floors passing: contrast
// rules are colour-agnostic, and the eye's optics are not.
//
// It measures three things, for every text/background pair in both modes:
//
//   1. WCAG ratio (the existing gate, listed for reference).
//   2. APCA Lc (apca-w3 0.1.9, vendored below). APCA models the eye's
//      light-adaptation and glare, so it and WCAG disagree on some pairs —
//      a pair can clear 4.5:1 and still read "dim" on a bright screen.
//      Floors: 60 = AA body text, 45 = large text / non-text, 75 = AAA.
//   3. Chromatic-aberration energy, our own measure. Blue/violet wavelengths
//      focus in front of the retina and yellow/red behind it, so a *cool* text
//      hue (roughly 200–340°) on a *warm* ground (55–140°) makes the lens hunt
//      between two focal planes at every letter edge. The pain scales with the
//      hue gap times the text colour's chroma:
//
//          energy = angularHueDistance(fg, bg) × C(fg)
//
//      The rule: a cool-on-warm text pair with energy >= 3.0 is a known eye
//      strain source. The body ink used to be violet #545464 on yellow paper —
//      173° apart at C 0.026, energy 4.5. #4f4f58 is the same hue at C 0.015,
//      energy 2.5, under the line. `--primary` links would flag it too, and do:
//      links are short, underlined and spaced, so the energy there is reported
//      but not treated as a failure.
//
// Each text pair also prints the smallest size its APCA contrast supports at
// weight 400, from APCA's own font-size table (reference face: Barlow). The
// app's 14–16px body needs a bare Lc around 70 at weight 400; anything under
// that reads as strain on a bright screen even with WCAG "passing" quietly.
//
// Self-tests run first and abort on mismatch, so a future transcription bug in
// the vendored block fails loudly instead of silently misreading the theme.

import { themeTokens } from "./theme-tokens.mjs"

///////////////////////////////////////////////////////////////////////////////
// Vendored from apca-w3 0.1.9 (Myndex / A. Somers), W3 license.
// https://github.com/Myndex/apca-w3 — SA98G, APCAcontrast(), sRGBtoY() and
// fontLookupAPCA() are verbatim; the source's unused declarations were dropped
// and its ternary-expression returns rewritten as if/else (identical math).
// The color-parsing wrapper (colorParsley) is not needed — this tool reads hex.
///////////////////////////////////////////////////////////////////////////////
const SA98G = {
  mainTRC: 2.4,
  get mainTRCencode() { return 1 / this.mainTRC },
  sRco: 0.2126729,
  sGco: 0.7151522,
  sBco: 0.0721750,
  normBG: 0.56,
  normTXT: 0.57,
  revTXT: 0.62,
  revBG: 0.65,
  blkThrs: 0.022,
  blkClmp: 1.414,
  scaleBoW: 1.14,
  scaleWoB: 1.14,
  loBoWoffset: 0.027,
  loWoBoffset: 0.027,
  deltaYmin: 0.0005,
  loClip: 0.1,
  mFactor: 1.94685544331710,
  get mFactInv() { return 1 / this.mFactor },
  mOffsetIn: 0.03873938165714010,
  mExpAdj: 0.2833433964208690,
  get mExp() { return this.mExpAdj / this.blkClmp },
  mOffsetOut: 0.3128657958707580,
}

function APCAcontrast(txtY, bgY, places = -1) {
  const icp = [0.0, 1.1]
  if (isNaN(txtY) || isNaN(bgY) || Math.min(txtY, bgY) < icp[0] || Math.max(txtY, bgY) > icp[1]) {
    return 0.0
  }
  let SAPC = 0.0
  let outputContrast = 0.0
  let polCat = "BoW"
  txtY = txtY > SA98G.blkThrs ? txtY : txtY + Math.pow(SA98G.blkThrs - txtY, SA98G.blkClmp)
  bgY = bgY > SA98G.blkThrs ? bgY : bgY + Math.pow(SA98G.blkThrs - bgY, SA98G.blkClmp)
  if (Math.abs(bgY - txtY) < SA98G.deltaYmin) {
    return 0.0
  }
  if (bgY > txtY) {
    SAPC = (Math.pow(bgY, SA98G.normBG) - Math.pow(txtY, SA98G.normTXT)) * SA98G.scaleBoW
    outputContrast = SAPC < SA98G.loClip ? 0.0 : SAPC - SA98G.loBoWoffset
  } else {
    polCat = "WoB"
    SAPC = (Math.pow(bgY, SA98G.revBG) - Math.pow(txtY, SA98G.revTXT)) * SA98G.scaleWoB
    outputContrast = SAPC > -SA98G.loClip ? 0.0 : SAPC + SA98G.loWoBoffset
  }
  if (places < 0) {
    return outputContrast * 100.0
  } else if (places == 0) {
    return Math.round(Math.abs(outputContrast) * 100.0) + "<sub>" + polCat + "</sub>"
  } else if (Number.isInteger(places)) {
    return (outputContrast * 100.0).toFixed(places)
  } else {
    return 0.0
  }
}

function sRGBtoY(rgb = [0, 0, 0]) {
  function simpleExp(chan) {
    return Math.pow(chan / 255.0, SA98G.mainTRC)
  }
  return (
    SA98G.sRco * simpleExp(rgb[0]) +
    SA98G.sGco * simpleExp(rgb[1]) +
    SA98G.sBco * simpleExp(rgb[2])
  )
}

const fontMatrixAscend = [
  ["Lc", 100, 200, 300, 400, 500, 600, 700, 800, 900],
  [0, 999, 999, 999, 999, 999, 999, 999, 999, 999],
  [10, 999, 999, 999, 999, 999, 999, 999, 999, 999],
  [15, 777, 777, 777, 777, 777, 777, 777, 777, 777],
  [20, 777, 777, 777, 777, 777, 777, 777, 777, 777],
  [25, 777, 777, 777, 120, 120, 108, 96, 96, 96],
  [30, 777, 777, 120, 108, 108, 96, 72, 72, 72],
  [35, 777, 120, 108, 96, 72, 60, 48, 48, 48],
  [40, 120, 108, 96, 60, 48, 42, 32, 32, 32],
  [45, 108, 96, 72, 42, 32, 28, 24, 24, 24],
  [50, 96, 72, 60, 32, 28, 24, 21, 21, 21],
  [55, 80, 60, 48, 28, 24, 21, 18, 18, 18],
  [60, 72, 48, 42, 24, 21, 18, 16, 16, 18],
  [65, 68, 46, 32, 21.75, 19, 17, 15, 16, 18],
  [70, 64, 44, 28, 19.5, 18, 16, 14.5, 16, 18],
  [75, 60, 42, 24, 18, 16, 15, 14, 16, 18],
  [80, 56, 38.25, 23, 17.25, 15.81, 14.81, 14, 16, 18],
  [85, 52, 34.5, 22, 16.5, 15.625, 14.625, 14, 16, 18],
  [90, 48, 32, 21, 16, 15.5, 14.5, 14, 16, 18],
  [95, 45, 28, 19.5, 15.5, 15, 14, 13.5, 16, 18],
  [100, 42, 26.5, 18.5, 15, 14.5, 13.5, 13, 16, 18],
  [105, 39, 25, 18, 14.5, 14, 13, 12, 16, 18],
  [110, 36, 24, 18, 14, 13, 12, 11, 16, 18],
  [115, 34.5, 22.5, 17.25, 12.5, 11.875, 11.25, 10.625, 14.5, 16.5],
  [120, 33, 21, 16.5, 11, 10.75, 10.5, 10.25, 13, 15],
  [125, 32, 20, 16, 10, 10, 10, 10, 12, 14],
]

const fontDeltaAscend = [
  ["∆Lc", 100, 200, 300, 400, 500, 600, 700, 800, 900],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [15, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [20, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [25, 0, 0, 0, 12, 12, 12, 24, 24, 24],
  [30, 0, 0, 12, 12, 36, 36, 24, 24, 24],
  [35, 0, 12, 12, 36, 24, 18, 16, 16, 16],
  [40, 12, 12, 24, 18, 16, 14, 8, 8, 8],
  [45, 12, 24, 12, 10, 4, 4, 3, 3, 3],
  [50, 16, 12, 12, 4, 4, 3, 3, 3, 3],
  [55, 8, 12, 6, 4, 3, 3, 2, 2, 0],
  [60, 4, 2, 10, 2.25, 2, 1, 1, 0, 0],
  [65, 4, 2, 4, 2.25, 1, 1, 0.5, 0, 0],
  [70, 4, 2, 4, 1.5, 2, 1, 0.5, 0, 0],
  [75, 4, 3.75, 1, 0.75, 0.188, 0.188, 0, 0, 0],
  [80, 4, 3.75, 1, 0.75, 0.188, 0.188, 0, 0, 0],
  [85, 4, 2.5, 1, 0.5, 0.125, 0.125, 0, 0, 0],
  [90, 3, 4, 1.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
  [95, 3, 1.5, 1, 0.5, 0.5, 0.5, 0.5, 0, 0],
  [100, 3, 1.5, 0.5, 0.5, 0.5, 0.5, 1, 0, 0],
  [105, 3, 1, 0, 0.5, 1, 1, 1, 0, 0],
  [110, 1.5, 1.5, 0.75, 1.5, 1.125, 0.75, 0.375, 1.5, 1.5],
  [115, 1.5, 1.5, 0.75, 1.5, 1.125, 0.75, 0.375, 1.5, 1.5],
  [120, 1, 1, 0.5, 1, 0.75, 0.5, 0.25, 1, 1],
  [125, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

const weightArray = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
const weightArrayLen = weightArray.length

function fontLookupAPCA(contrast, places = 2) {
  let returnArray = [contrast.toFixed(places), 0, 0, 0, 0, 0, 0, 0, 0, 0]
  contrast = Math.abs(contrast)
  const factor = 0.2
  const index = contrast == 0 ? 1 : (contrast * factor) | 0
  let w = 0
  let scoreAdj = (contrast - fontMatrixAscend[index][w]) * factor
  w++
  let tempFont = 777
  for (; w < weightArrayLen; w++) {
    tempFont = fontMatrixAscend[index][w]
    if (tempFont > 400) {
      returnArray[w] = tempFont
    } else if (contrast < 14.5) {
      returnArray[w] = 999
    } else if (contrast < 29.5) {
      returnArray[w] = 777
    } else {
      if (tempFont > 24) {
        returnArray[w] = Math.round(tempFont - fontDeltaAscend[index][w] * scoreAdj)
      } else {
        returnArray[w] = tempFont - ((2.0 * fontDeltaAscend[index][w] * scoreAdj) | 0) * 0.5
      }
    }
  }
  return returnArray
}
// ── End vendored apca-w3 (0.1.9, W3 license) ────────────────────────────────

// ---- CSS tokens (resolved through the three layers, as the browser would) ---
const { light, dark } = themeTokens()

// ---- Colour math -----------------------------------------------------------
const srgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16))
const lin = (c) => {
  c /= 255
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
}
const Y = (h) => {
  const [r, g, b] = srgb(h).map(lin)
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}
const ratio = (a, b) => {
  const [x, y] = [Y(a), Y(b)].sort((p, q) => q - p)
  return (x + 0.05) / (y + 0.05)
}

// OKLab / OKLCH (Björn Ottosson's space — hue here is what the CA measure cares
// about, since it is the perceptual hue, not sRGB's).
const RGBtoLMS = [
  [0.4122214708, 0.5363325363, 0.0514459929],
  [0.2119034982, 0.6806995451, 0.1073969566],
  [0.0883024619, 0.2817188376, 0.6299787005],
]
const LMStoLab = [
  [0.2104542553, 0.7936177850, -0.0040720468],
  [1.9779984951, -2.4285922050, 0.4505937099],
  [0.0259040371, 0.7827717662, -0.8086757660],
]
const croot = (x) => (x < 0 ? -Math.cbrt(-x) : Math.cbrt(x))
const oklch = (h) => {
  const l = srgb(h).map(lin)
  const m = RGBtoLMS.map((row) => row.reduce((s, v, j) => s + v * l[j], 0)).map(croot)
  let L = 0, a = 0, b = 0
  for (let i = 0; i < 3; i++) {
    L += LMStoLab[0][i] * m[i]
    a += LMStoLab[1][i] * m[i]
    b += LMStoLab[2][i] * m[i]
  }
  return { L, C: Math.hypot(a, b), H: (((Math.atan2(b, a) * 180) / Math.PI + 360) % 360) }
}
const hueDist = (a, b) => {
  const d = Math.abs(a - b) % 360
  return Math.min(d, 360 - d)
}
// Chromatic-aberration energy of a text pair. Only meaningful when the text is
// on the *short-wavelength* side of the wheel and the ground on the long side —
// a warm gray on warm paper is fine no matter its chroma.
const CA = (fg, bg) => {
  const f = oklch(fg)
  const g = oklch(bg)
  const coolOnWarm = f.H >= 200 && f.H <= 340 && g.H >= 55 && g.H <= 140
  return { energy: hueDist(f.H, g.H) * f.C, coolOnWarm }
}
const CA_LIMIT = 3.0

// fontLookupAPCA returns [contrast, w100, w200, … w900] — the weight columns of
// fontMatrixAscend, whose row is ["Lc", 100, 200, …]. So weight 400 is index 4,
// and reading index 5 would report the smallest size at weight *500* instead:
// lighter than the app sets its body, and so a size that looks safe and is not.
const W400 = 4

// ---- Self-tests ------------------------------------------------------------
function check(name, got, want, tol = 0.3) {
  const ok = Math.abs(got - want) <= tol
  console.log(`  ${ok ? "ok  " : "FAIL"} ${name.padEnd(26)} ${got.toFixed(2)} (want ${want})`)
  if (!ok) process.exitCode = 1
}
console.log("── self-tests (abort-worthy) ─────────────────────────────────")
check("APCA #000 on #fff", APCAcontrast(sRGBtoY([0, 0, 0]), sRGBtoY([255, 255, 255])), 106.04)
check("APCA #888 on #fff", APCAcontrast(sRGBtoY([136, 136, 136]), sRGBtoY([255, 255, 255])), 63.06)
check("APCA #fff on #000", APCAcontrast(sRGBtoY([255, 255, 255]), sRGBtoY([0, 0, 0])), -107.88)
check("OKLCH #fff L", oklch("#ffffff").L, 1.0, 0.001)
check("OKLCH #000 L", oklch("#000000").L, 0.0, 0.001)
if (process.exitCode) process.exit(1)

// ---- Output ----------------------------------------------------------------
// [label, fg, bg, isBody] — the pairs that *carry* text (floor 4.5 in WCAG).
// `isBody` marks the long-form tokens; short-form colours (links, chips) are
// reported but never passed or failed — the CA column there is information.
const pairs = (t) => [
  ["foreground / background", t.foreground, t.background, true],
  ["foreground / card", t["card-foreground"], t.card, true],
  ["foreground / popover", t["popover-foreground"], t.popover, true],
  ["foreground / secondary", t["secondary-foreground"], t.secondary, true],
  ["foreground / accent", t["accent-foreground"], t.accent, true],
  ["foreground / muted", t.foreground, t.muted, true],
  ["muted-fg / background", t["muted-foreground"], t.background, true],
  ["muted-fg / card", t["muted-foreground"], t.card, true],
  ["muted-fg / muted", t["muted-foreground"], t.muted, true],
  ["muted-fg / secondary", t["muted-foreground"], t.secondary, true],
  ["sidebar-fg / sidebar", t["sidebar-foreground"], t.sidebar, true],
  ["sidebar-accent-fg / accent", t["sidebar-accent-foreground"], t["sidebar-accent"], true],
  ["primary (links) / paper", t.primary, t.background, false],
  ["primary / card", t.primary, t.card, false],
  ["primary-fg / primary", t["primary-foreground"], t.primary, false],
  ["destructive / paper", t.destructive, t.background, false],
]

console.log("\n──── LIGHT (warm paper) token ladder ─────────────────────────")
console.log("  token".padEnd(24) + "hex        Y      L      C     H ")
for (const [k, v] of Object.entries(light).sort()) {
  const l = oklch(v)
  console.log(
    `  ${k.padEnd(22)}${v}  ${Y(v).toFixed(3).padStart(5)}  ${l.L.toFixed(3).padStart(6)}  ${l.C.toFixed(3).padStart(6)}  ${l.H.toFixed(1).padStart(6)}`
  )
}

console.log("\n──── LIGHT text pairs ────────────────────────────────────────")
console.log("  pair".padEnd(34) + "WCAG   APCA   font400  CA-energy  ")
for (const [label, fg, bg, isBody] of pairs(light).filter(([, f, b]) => f && b)) {
  const w = ratio(fg, bg)
  const lc = APCAcontrast(sRGBtoY(srgb(fg)), sRGBtoY(srgb(bg)))
  const font = fontLookupAPCA(Math.min(lc, 125))
  const { energy, coolOnWarm } = CA(fg, bg)
  const note = isBody
    ? coolOnWarm
      ? energy >= CA_LIMIT
        ? "  CA! cool text × warm ground, over the line"
        : `  cool × warm — energy ${energy.toFixed(1)}, under the line`
      : "  no cool×warm fringe"
    : ""
  console.log(
    `  ${label.padEnd(32)} ${w.toFixed(2).padStart(5)} ${lc.toFixed(1).padStart(6)} ${font[W400].toString().padStart(7)}  ${energy.toFixed(2).padStart(6)}  ${note}`
  )
}

console.log("\n──── DARK (dragon) token ladder ──────────────────────────────")
console.log("  token".padEnd(24) + "hex        Y      L      C     H ")
for (const [k, v] of Object.entries(dark).sort()) {
  const l = oklch(v)
  console.log(`  ${k.padEnd(22)}${v}  ${Y(v).toFixed(3).padStart(5)}  ${l.L.toFixed(3).padStart(6)}  ${l.C.toFixed(3).padStart(6)}  ${l.H.toFixed(1).padStart(6)}`)
}

console.log("\n──── DARK text pairs ─────────────────────────────────────────")
console.log("  pair".padEnd(34) + "WCAG   APCA   font400  CA-energy  ")
for (const [label, fg, bg, isBody] of pairs(dark).filter(([, f, b]) => f && b)) {
  const w = ratio(fg, bg)
  const lc = APCAcontrast(sRGBtoY(srgb(fg)), sRGBtoY(srgb(bg)))
  const font = fontLookupAPCA(Math.min(lc, 125))
  const { energy, coolOnWarm } = CA(fg, bg)
  const note = isBody
    ? coolOnWarm
      ? energy >= CA_LIMIT
        ? "  CA! cool text × warm ground, over the line"
        : `  cool × warm — energy ${energy.toFixed(1)}, under the line`
      : "  no cool×warm fringe"
    : ""
  console.log(`  ${label.padEnd(32)} ${w.toFixed(2).padStart(5)} ${lc.toFixed(1).padStart(6)} ${font[W400].toString().padStart(7)}  ${energy.toFixed(2).padStart(6)}  ${note}`)
}