import type { ResourceSummary, RevisionRef } from "./contracts.generated"
import { decodeArchitecture } from "./decoder"

/** Formats a deliberately selected immutable module pin for manual YAML adoption; archived or foreign revisions cannot produce a new-selection snippet. */
export function dependencySnippet(
  instance: string,
  resource: ResourceSummary,
  revision: RevisionRef
): string {
  if (!/^[a-z][a-z0-9_-]{0,63}$/.test(instance))
    throw new Error(
      "Use a lowercase module name of at most 64 letters, digits, hyphens or underscores."
    )
  const reference = decodeArchitecture("RevisionRef", revision)
  if (
    resource.archived ||
    reference.resource_id !== resource.resource_id ||
    reference.kind !== resource.kind
  )
    throw new Error(
      "Choose an unarchived resource and one of its accepted revisions."
    )
  return `modules:\n  ${instance}:\n    reference: ${JSON.stringify(reference)}\n    parameters: {}\n`
}
