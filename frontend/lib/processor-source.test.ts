import assert from "node:assert/strict"
import { test } from "node:test"
import { readFileSync } from "node:fs"

import { parseProcessorCatalogue, parseProcessorSource } from "./data"

const contract = JSON.parse(
  readFileSync(
    new URL("../../contracts/processor-source.json", import.meta.url),
    "utf8"
  )
)
const status = contract.source_response

test("the shared catalogue contract validates active hashes and rejects malformed forms", () => {
  assert.deepEqual(
    parseProcessorCatalogue(contract.catalogue),
    contract.catalogue.processors
  )
  assert.throws(() =>
    parseProcessorCatalogue({
      processors: [{ ...contract.catalogue.processors[0], version_hash: null }],
    })
  )
  assert.throws(() =>
    parseProcessorCatalogue({
      processors: [{ ...contract.catalogue.processors[0], params: [{}] }],
    })
  )
})

test("source responses retain saved/active divergence and literal invalid source", () => {
  assert.deepEqual(parseProcessorSource(status), status)
})

test("the editor rejects malformed source identity, hashes, editability and load errors", () => {
  for (const invalid of [
    null,
    {},
    { ...status, version_hash: "wrong" },
    { ...status, source: 1 },
    { ...status, editable: "true" },
    { ...status, reload_error: {} },
  ]) {
    assert.throws(() => parseProcessorSource(invalid))
  }
})
