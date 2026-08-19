// A line diff, for showing what a cleaning step did to a document.
//
// Small on purpose. This compares two versions of one document in a preview —
// hundreds of lines, not a repository — so the classic LCS table is the right
// algorithm and a dependency would be the wrong trade. Above `MAX_LINES` the
// table stops being cheap, and the pair is reported as one wholesale
// replacement rather than quietly costing seconds.

/** What happened to one line, in the split view's vocabulary. */
export type RowKind = "same" | "removed" | "added" | "changed"

/**
 * One row of the split view: the left side, the right side, or both.
 *
 * `null` on a side means that side has no line there — the gap GitHub renders
 * as an empty cell, which is what makes the two columns stay in step.
 */
export interface DiffRow {
  kind: RowKind
  left: { number: number; text: string } | null
  right: { number: number; text: string } | null
}

/** A run of unchanged rows the view may fold away, and where it sits. */
export interface DiffHunk {
  rows: DiffRow[]
  /** True when every row is unchanged and the run is long enough to collapse. */
  collapsible: boolean
}

/** Beyond this the LCS table is no longer worth building. */
const MAX_LINES = 600

/** Unchanged lines kept either side of a change, as context. */
const CONTEXT = 2

/** A shorter run than this is cheaper to show than to fold. */
const MIN_FOLD = 6

/** Rows for a split view, aligned so both columns advance together. */
export function splitDiff(before: string, after: string): DiffRow[] {
  const left = before.length === 0 ? [] : before.split("\n")
  const right = after.length === 0 ? [] : after.split("\n")

  if (left.length > MAX_LINES || right.length > MAX_LINES) {
    return pairUp(
      left.map((text, i) => ({ number: i + 1, text })),
      right.map((text, i) => ({ number: i + 1, text }))
    )
  }

  return align(left, right)
}

/**
 * Group rows into hunks, marking long unchanged runs as foldable.
 *
 * A cleaning step usually touches a few lines of a long document, and a view
 * that renders every untouched line buries the answer to "what did this do".
 */
export function hunks(rows: DiffRow[]): DiffHunk[] {
  const interesting = new Set<number>()
  rows.forEach((row, i) => {
    if (row.kind === "same") return
    for (let j = i - CONTEXT; j <= i + CONTEXT; j++) interesting.add(j)
  })

  const grouped: DiffHunk[] = []
  for (const [i, row] of rows.entries()) {
    const collapsible = !interesting.has(i)
    const last = grouped.at(-1)
    if (last && last.collapsible === collapsible) last.rows.push(row)
    else grouped.push({ rows: [row], collapsible })
  }

  // A two-line fold saves nothing and costs a click.
  for (const hunk of grouped) {
    if (hunk.rows.length < MIN_FOLD) hunk.collapsible = false
  }
  return grouped
}

type Line = { number: number; text: string }

/** Longest common subsequence over lines, walked back into rows. */
function align(left: string[], right: string[]): DiffRow[] {
  const table = lcs(left, right)

  const rows: DiffRow[] = []
  let removed: Line[] = []
  let added: Line[] = []
  let i = 0
  let j = 0

  const flush = () => {
    if (removed.length || added.length) rows.push(...pairUp(removed, added))
    removed = []
    added = []
  }

  while (i < left.length && j < right.length) {
    if (left[i] === right[j]) {
      flush()
      rows.push({
        kind: "same",
        left: { number: i + 1, text: left[i] },
        right: { number: j + 1, text: right[j] },
      })
      i++
      j++
    } else if (table[i + 1][j] >= table[i][j + 1]) {
      removed.push({ number: i + 1, text: left[i] })
      i++
    } else {
      added.push({ number: j + 1, text: right[j] })
      j++
    }
  }
  while (i < left.length) removed.push({ number: i + 1, text: left[i++] })
  while (j < right.length) added.push({ number: j + 1, text: right[j++] })
  flush()

  return rows
}

/** The classic table, built backwards so the walk above reads forwards. */
function lcs(left: string[], right: string[]): number[][] {
  const table = Array.from({ length: left.length + 1 }, () =>
    new Array<number>(right.length + 1).fill(0)
  )
  for (let i = left.length - 1; i >= 0; i--) {
    for (let j = right.length - 1; j >= 0; j--) {
      table[i][j] =
        left[i] === right[j]
          ? table[i + 1][j + 1] + 1
          : Math.max(table[i + 1][j], table[i][j + 1])
    }
  }
  return table
}

/**
 * A block of removals and a block of additions, set opposite each other.
 *
 * Lines that pair up read as one edited line rather than as a deletion followed
 * by an unrelated insertion — which is what a whitespace or redaction step
 * actually did to them.
 */
function pairUp(removed: Line[], added: Line[]): DiffRow[] {
  const rows: DiffRow[] = []
  const paired = Math.min(removed.length, added.length)

  for (let k = 0; k < paired; k++) {
    rows.push({ kind: "changed", left: removed[k], right: added[k] })
  }
  for (let k = paired; k < removed.length; k++) {
    rows.push({ kind: "removed", left: removed[k], right: null })
  }
  for (let k = paired; k < added.length; k++) {
    rows.push({ kind: "added", left: null, right: added[k] })
  }
  return rows
}
