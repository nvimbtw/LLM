import { execFile } from "node:child_process"
import { promisify } from "node:util"
import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs"
import { resolve } from "node:path"
import { artifactHashes, root } from "./throbber-artifacts.mjs"

const execute = promisify(execFile)
const version = (await execute("wasm-bindgen", ["--version"])).stdout.trim()
if (version !== "wasm-bindgen 0.2.125")
  throw new Error("Install wasm-bindgen-cli 0.2.125 to match Cargo.lock")
const build = await execute(
  "cargo",
  [
    "build",
    "--locked",
    "--release",
    "--target",
    "wasm32-unknown-unknown",
    "--manifest-path",
    "gpu/Cargo.toml",
  ],
  { cwd: root }
)
process.stderr.write(build.stderr)
const target = process.env.CARGO_TARGET_DIR ?? resolve(root, "gpu/target")
await execute(
  "wasm-bindgen",
  [
    resolve(target, "wasm32-unknown-unknown/release/slate_throbber.wasm"),
    "--target",
    "web",
    "--out-dir",
    "lib/live-throbber/generated",
    "--out-name",
    "renderer",
  ],
  { cwd: root }
)
// Bundlers resolve new URL(..., import.meta.url) even when init gets an explicit path.
const loaderPath = resolve(root, "lib/live-throbber/generated/renderer.js")
const loader = readFileSync(loaderPath, "utf8")
const defaultUrl = "new URL('renderer_bg.wasm', import.meta.url)"
if (!loader.includes(defaultUrl))
  throw new Error("Unexpected wasm-bindgen loader format")
writeFileSync(
  loaderPath,
  loader.replace(defaultUrl, '"/live-throbber/renderer_bg.wasm"')
)
mkdirSync(resolve(root, "public/live-throbber"), { recursive: true })
renameSync(
  resolve(root, "lib/live-throbber/generated/renderer_bg.wasm"),
  resolve(root, "public/live-throbber/renderer_bg.wasm")
)
writeFileSync(
  resolve(root, "gpu/assets.json"),
  JSON.stringify(artifactHashes(), null, 2) + "\n"
)
