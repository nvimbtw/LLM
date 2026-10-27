import Ajv from "ajv/dist/2020"
import { sha256 } from "@noble/hashes/sha256"
import { bytesToHex } from "@noble/hashes/utils"
import {
  architectureSchemas,
  type ArchitectureContracts,
} from "./contracts.generated"
import { checkSemantics } from "./semantics"
export type ContractName = keyof ArchitectureContracts
export type JsonRecord = Record<string, unknown>
type Schema = JsonRecord & {
  $defs?: Record<string, Schema>
  $ref?: string
  properties?: Record<string, Schema>
  items?: Schema
  additionalProperties?: boolean | Schema
  anyOf?: Schema[]
  oneOf?: Schema[]
  patternProperties?: Record<string, Schema>
}
const ajv = new Ajv({ strict: false, strictNumbers: true, allErrors: true })
ajv.addFormat(
  "uuid",
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
)
const validators = new Map<ContractName, ReturnType<typeof ajv.compile>>()
const branchValidators = new WeakMap<Schema, ReturnType<typeof ajv.compile>>()

/** Reuses branch validators for the lifetime of immutable imported schema declarations. */
function matchesBranch(
  candidate: Schema,
  root: Schema,
  value: unknown
): boolean {
  let validate = branchValidators.get(candidate)
  if (!validate) {
    validate = ajv.compile({ ...candidate, $defs: root.$defs })
    branchValidators.set(candidate, validate)
  }
  return Boolean(validate(value))
}

/** Hashes normalized JSON with sorted keys and compact UTF-8 encoding. */
export function identityDigest(value: unknown): string {
  return bytesToHex(
    sha256(
      new TextEncoder().encode(
        JSON.stringify(value, (_, item) =>
          item && typeof item === "object" && !Array.isArray(item)
            ? Object.fromEntries(
                Object.keys(item)
                  .sort()
                  .map((key) => [key, item[key]])
              )
            : item
        )
      )
    )
  )
}
/** Applies declaration defaults to a private snapshot and checks nested semantics. */
function normalize(value: unknown, schema: Schema, root: Schema): unknown {
  if (schema.$ref)
    return normalize(value, root.$defs![schema.$ref.split("/").at(-1)!], root)
  const branches = schema.anyOf ?? schema.oneOf
  if (branches) {
    const branch = branches.find((candidate) =>
      matchesBranch(candidate, root, value)
    )
    return branch ? normalize(value, branch, root) : value
  }
  if (Array.isArray(value) && schema.items)
    return value.map((item) => normalize(item, schema.items!, root))
  if (!value || typeof value !== "object" || Array.isArray(value)) return value
  const record: JsonRecord = { ...value }
  for (const [key, property] of Object.entries(schema.properties ?? {})) {
    if (!(key in record) && "default" in property)
      record[key] = structuredClone(property.default)
    if (
      !(key in record) &&
      !(schema.required as string[] | undefined)?.includes(key)
    ) {
      if (property.type === "object") record[key] = {}
      if (property.type === "array") record[key] = []
    }
    if (key in record) record[key] = normalize(record[key], property, root)
  }
  if (typeof schema.additionalProperties === "object")
    for (const key of Object.keys(record))
      record[key] = normalize(record[key], schema.additionalProperties, root)
  for (const [pattern, property] of Object.entries(
    schema.patternProperties ?? {}
  )) {
    for (const key of Object.keys(record)) {
      if (!new RegExp(pattern).test(key))
        throw new Error("Invalid declared identifier")
      record[key] = normalize(record[key], property, root)
    }
  }
  checkSemantics(String(schema.title ?? ""), record)
  return record
}
/** Validates interpreted wire data using the authoritative shared schema and semantic rules. */
export function decodeArchitecture<K extends ContractName>(
  name: K,
  value: unknown
): ArchitectureContracts[K] {
  let validate = validators.get(name)
  if (!validate) {
    validate = ajv.compile(architectureSchemas[name])
    validators.set(name, validate)
  }
  if (!validate(value))
    throw new Error(`Invalid ${name}: ${ajv.errorsText(validate.errors)}`)
  return normalize(
    value,
    architectureSchemas[name] as Schema,
    architectureSchemas[name] as Schema
  ) as ArchitectureContracts[K]
}

/** Validates outgoing values while preserving omitted fields required by optimistic patch semantics. */
export function encodeArchitecture<K extends ContractName>(
  name: K,
  value: ArchitectureContracts[K]
): string {
  decodeArchitecture(name, value)
  return JSON.stringify(value)
}
