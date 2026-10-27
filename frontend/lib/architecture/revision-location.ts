export interface RevisionLocationQuery { revision?: string | string[]; line?: string | string[] }
export interface RevisionLocation { revision?: string; line?: number }

/** Interprets an exact immutable-source navigation target, refusing malformed/ambiguous queries instead of opening a different source as fallback. */
export function revisionLocation(query: RevisionLocationQuery): RevisionLocation | null {
  if (query.revision === undefined) return query.line === undefined ? {} : null
  if (typeof query.revision !== "string" || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(query.revision)) return null
  if (query.line !== undefined && (typeof query.line !== "string" || !/^[1-9][0-9]{0,6}$/.test(query.line) || Number(query.line) > 1_000_000)) return null
  return { revision: query.revision.toLowerCase(), ...(query.line === undefined ? {} : { line: Number(query.line) }) }
}
