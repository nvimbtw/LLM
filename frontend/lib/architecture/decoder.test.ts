import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { requireCurrentReceipt } from "../model-validation"
import { requireGraphCursor } from "../model-graph"
import { decodeArchitecture, type ContractName } from "./decoder"
for (const group of ["api", "sdk"]) {
  const root = new URL(
    `../../../contracts/modular-architecture/v1/${group}/`,
    import.meta.url
  )
  const fixtures = JSON.parse(readFileSync(new URL("cases.json", root), "utf8"))
  for (const fixture of fixtures.cases) {
    if (["DefinitionPackage", "ResolvedDefinition"].includes(fixture.model))
      continue
    test(`shared ${group}: ${fixture.name}`, () => {
      const value = fixture.path
        ? JSON.parse(readFileSync(new URL(fixture.path, root), "utf8"))
        : fixture.value
      const decode = () => {
        const result = decodeArchitecture(fixture.model as ContractName, value)
        if (fixture.check === "current_receipt")
          requireCurrentReceipt(
            decodeArchitecture("ValidationReceipt", value),
            decodeArchitecture("ReceiptIdentity", fixture.context.expected)
          )
        if (fixture.check === "graph_cursor")
          requireGraphCursor(
            decodeArchitecture("GraphCursor", value),
            decodeArchitecture("GraphQuery", fixture.context.query),
            fixture.context.graph_hash
          )
        return result
      }
      if (fixture.valid) assert.doesNotThrow(decode)
      else assert.throws(decode)
    })
  }
}
