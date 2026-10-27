import type { ParameterDeclaration } from "./contracts.generated"
export type ParameterValue = string | number | boolean
/** Checks one declared constructor value without coercion, preserving enum identity and numeric bounds. */
export function parameterError(
  declaration: ParameterDeclaration,
  value: unknown
): string | null {
  const type = declaration.type
  if (
    type === "integer" &&
    (typeof value !== "number" || !Number.isSafeInteger(value))
  )
    return "Enter an exact integer."
  if (
    type === "number" &&
    (typeof value !== "number" || !Number.isFinite(value))
  )
    return "Enter a finite number."
  if (type === "boolean" && typeof value !== "boolean")
    return "Choose true or false."
  if (
    type === "string" &&
    (typeof value !== "string" || Array.from(value).length > 4096)
  )
    return "Enter at most 4096 characters."
  if (
    type === "enum" &&
    !declaration.choices?.some(
      (choice) => typeof choice === typeof value && choice === value
    )
  )
    return "Choose a declared value."
  if (
    declaration.minimum != null &&
    (typeof value !== "number" || value < declaration.minimum)
  )
    return `Minimum: ${declaration.minimum}.`
  if (
    declaration.maximum != null &&
    (typeof value !== "number" || value > declaration.maximum)
  )
    return `Maximum: ${declaration.maximum}.`
  return null
}
