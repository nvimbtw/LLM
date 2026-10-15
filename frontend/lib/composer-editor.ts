import { Extension, Mark, type Editor } from "@tiptap/core"
import type { ResolvedPos } from "@tiptap/pm/model"
import { AllSelection } from "@tiptap/pm/state"

const CODE_INDENT = "  "

/** Deletes a fully selected draft through its selection contract, leaving a caret in the empty paragraph and preserving undo. Other selections use the standard editor keys. */
function deleteSelectedDocument(editor: Editor): boolean {
  if (!(editor.state.selection instanceof AllSelection)) return false
  editor.view.dispatch(editor.state.tr.deleteSelection().scrollIntoView())
  return true
}

/**
 * Indents/outdents code lines touched by the selection, returning false outside code blocks.
 * Clamps selections to the block and edits backwards so earlier document positions remain valid.
 */
function indentCode(editor: Editor, outdent: boolean): boolean {
  const { state } = editor
  const { from, to, empty } = state.selection

  const $from = state.doc.resolve(from)
  const $to = state.doc.resolve(to)
  const $in = codeBlockDepth($from) !== null ? $from : $to
  const depth = codeBlockDepth($in)
  if (depth === null) return false

  const start = $in.start(depth)
  const end = $in.end(depth)
  const text = $in.node(depth).textContent

  if (empty && !outdent) {
    editor.view.dispatch(state.tr.insertText(CODE_INDENT, from))
    return true
  }

  const selFrom = Math.max(from, start)
  const selTo = Math.min(to, end)

  const lineStarts: number[] = []
  for (let offset = 0; ; ) {
    const nl = text.indexOf("\n", offset)
    const lineEnd = nl === -1 ? text.length : nl
    if (start + lineEnd >= selFrom && start + offset <= selTo) {
      lineStarts.push(start + offset)
    }
    if (nl === -1) break
    offset = nl + 1
  }

  const tr = state.tr
  for (const pos of lineStarts.reverse()) {
    if (!outdent) {
      tr.insertText(CODE_INDENT, pos)
      continue
    }
    const lead = text.slice(pos - start).match(/^(\t| {1,2})/)
    if (lead) tr.delete(pos, pos + lead[1].length)
  }

  if (tr.docChanged) editor.view.dispatch(tr)
  return true
}

/** Returns the containing code-block depth for a resolved document position, or null outside code. */
function codeBlockDepth($pos: ResolvedPos): number | null {
  for (let d = $pos.depth; d > 0; d--) {
    if ($pos.node(d).type.name === "codeBlock") return d
  }
  return null
}

export const ComposerKeys = Extension.create({
  name: "composerKeys",
  priority: 1000,
  /** Supplies full-draft deletion and list/code indentation keys, consuming Tab even when no indentation action is available. */
  addKeyboardShortcuts() {
    return {
      Backspace: () => deleteSelectedDocument(this.editor),
      Delete: () => deleteSelectedDocument(this.editor),
      Tab: () =>
        indentCode(this.editor, false) ||
        this.editor.commands.sinkListItem("listItem") ||
        true,
      "Shift-Tab": () =>
        indentCode(this.editor, true) ||
        this.editor.commands.liftListItem("listItem") ||
        true,
    }
  },
})

export const InterimTranscript = Mark.create({
  name: "interimTranscript",
  inclusive: false,
  /** Renders the recognizer's revisable tail with reduced opacity; the non-inclusive mark never spreads onto typed text. */
  renderHTML() {
    return ["span", { class: "opacity-55" }, 0]
  },
})

export const VOICE_TRANSACTION = "voiceTranscript"
