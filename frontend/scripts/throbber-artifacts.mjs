import { createHash } from "node:crypto"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { resolve } from "node:path"

export const root = fileURLToPath(new URL("..", import.meta.url))
const sources = [
  "gpu/Cargo.toml",
  "gpu/Cargo.lock",
  "gpu/src/lib.rs",
  "gpu/src/art.wgsl",
  "scripts/build-throbber.mjs",
  "scripts/throbber-artifacts.mjs",
]
const artifacts = [
  "lib/live-throbber/generated/renderer.js",
  "lib/live-throbber/generated/renderer.d.ts",
  "lib/live-throbber/generated/renderer_bg.wasm.d.ts",
  "public/live-throbber/renderer_bg.wasm",
]

export function artifactHashes() {
  return Object.fromEntries(
    [...sources, ...artifacts].map((name) => [
      name,
      createHash("sha256")
        .update(readFileSync(resolve(root, name)))
        .digest("hex"),
    ])
  )
}
