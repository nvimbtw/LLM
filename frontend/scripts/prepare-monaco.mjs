import { cp, mkdir, rm } from "node:fs/promises"

const destination = new URL("../public/monaco/", import.meta.url)
await mkdir(destination, { recursive: true })
await rm(new URL("vs/", destination), { recursive: true, force: true })
await cp(
  new URL("../node_modules/monaco-editor/min/vs/", import.meta.url),
  new URL("vs/", destination),
  { recursive: true }
)
await cp(
  new URL("../node_modules/monaco-editor/LICENSE", import.meta.url),
  new URL("LICENSE", destination)
)
