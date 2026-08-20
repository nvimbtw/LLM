// The theme's tokens, resolved out of `app/globals.css` for both modes.
//
// The stylesheet is three layers deep — palette (`--kanagawa-*`, hexes), theme
// (`--role-light` / `--role-dark`, pointing at the palette), active mode
// (`--role`, pointing at one half of the theme) — so a token is no longer a hex
// sitting in a block a regex can lift. This resolves the chain the way the
// browser does and hands the audits what they have always wanted: one flat
// `{ role: "#rrggbb" }` per mode.
//
// The rules are walked in file order and applied to a scope the way the cascade
// would: a `:root` rule feeds both modes, `.theme-light` only the light one,
// `.dark` only the dark one. Anything that is not a plain hex or a single
// `var(--x)` reference (a `color-mix`, a gradient) is left out — the audits
// measure tokens, and a token that is not a colour has nothing to measure.

import { readFileSync } from "node:fs"

const HEX = /^#[0-9a-fA-F]{6}$/
const VAR = /^var\(\s*(--[\w-]+)\s*\)$/

/** Comments out of the way first: this stylesheet is more comment than CSS, and
 *  the text before a `{` is otherwise the block comment above the rule. */
const uncomment = (css) => css.replace(/\/\*[\s\S]*?\*\//g, "")

/** Split a stylesheet into top-level `selector { …declarations… }` rules. */
function rules(source) {
  const css = uncomment(source)
  const out = []
  let depth = 0
  let start = 0
  let selector = ""
  for (let i = 0; i < css.length; i++) {
    const c = css[i]
    if (c === "{") {
      if (depth === 0) {
        selector = css.slice(start, i).trim()
        start = i + 1
      }
      depth++
    } else if (c === "}") {
      depth--
      if (depth === 0) {
        out.push([selector, css.slice(start, i)])
        start = i + 1
      }
    }
  }
  return out
}

/** `--name: value;` pairs, ignoring anything nested or commented out. */
function declarations(body) {
  const out = []
  for (const m of body.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) {
    out.push([m[1], m[2].trim()])
  }
  return out
}

/**
 * `{ light, dark }` — each a `{ tokenName: "#rrggbb" }` map, token names
 * without their leading dashes so `t.foreground` and `t["muted-foreground"]`
 * read the way they do in the components.
 */
export function themeTokens(path = "app/globals.css") {
  const css = readFileSync(path, "utf8")
  const scopes = { light: new Map(), dark: new Map() }

  for (const [selector, body] of rules(css)) {
    // `@theme inline` and the like: not a colour scope.
    if (selector.startsWith("@")) continue
    const parts = selector.split(",").map((s) => s.trim())
    const targets = []
    if (parts.includes(":root")) targets.push("light", "dark")
    if (parts.includes(".theme-light")) targets.push("light")
    if (parts.includes(".dark")) targets.push("dark")
    if (!targets.length) continue
    for (const [name, value] of declarations(body)) {
      for (const t of targets) scopes[t].set(name, value)
    }
  }

  const resolve = (scope, name, seen = new Set()) => {
    if (seen.has(name)) return undefined // a cycle: report nothing rather than hang
    seen.add(name)
    const value = scope.get(name)
    if (!value) return undefined
    if (HEX.test(value)) return value.toLowerCase()
    const ref = value.match(VAR)
    return ref ? resolve(scope, ref[1], seen) : undefined
  }

  const flatten = (scope) => {
    const out = {}
    for (const name of scope.keys()) {
      // The palette and the per-mode halves are the machinery, not the tokens:
      // the audits speak in roles (`--foreground`), which is what is left.
      if (name.startsWith("--kanagawa-")) continue
      if (name.endsWith("-light") || name.endsWith("-dark")) continue
      const hex = resolve(scope, name)
      if (hex) out[name.slice(2)] = hex
    }
    return out
  }

  return { light: flatten(scopes.light), dark: flatten(scopes.dark) }
}

/** The palette itself, `{ "--kanagawa-…": "#rrggbb" }`, for tools that list it. */
export function palette(path = "app/globals.css") {
  const css = readFileSync(path, "utf8")
  const out = {}
  for (const [selector, body] of rules(css)) {
    if (!selector.split(",").map((s) => s.trim()).includes(":root")) continue
    for (const [name, value] of declarations(body)) {
      if (name.startsWith("--kanagawa-") && HEX.test(value)) out[name] = value.toLowerCase()
    }
  }
  return out
}
