import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { artifactHashes, root } from "./throbber-artifacts.mjs"

const expected = JSON.parse(
  readFileSync(resolve(root, "gpu/assets.json"), "utf8")
)
for (const [name, hash] of Object.entries(artifactHashes())) {
  if (expected[name] !== hash)
    throw new Error(
      `Stale throbber artifact: ${name}. Run bun run build:throbber.`
    )
}
console.log("Throbber WASM assets match their source.")
