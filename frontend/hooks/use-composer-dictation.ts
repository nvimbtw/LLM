"use client"

import * as React from "react"
import type { Editor, JSONContent } from "@tiptap/core"
import type { Transaction } from "@tiptap/pm/state"

import { useSpeechRecognition } from "./use-speech-recognition"
import { InterimTranscript, VOICE_TRANSACTION } from "@/lib/composer-editor"

interface VoiceRange {
  from: number
  to: number
}

/** Turns final/interim recognition strings into one editor span, marking only the revisable tail. */
function transcriptContent(final: string, interim: string): JSONContent[] {
  return [
    ...(final ? [{ type: "text", text: final }] : []),
    ...(interim
      ? [
          {
            type: "text",
            text: final ? ` ${interim}` : interim,
            marks: [{ type: InterimTranscript.name }],
          },
        ]
      : []),
  ]
}

/**
 * Replaces a dictation span in the supplied editor, returning its new document range.
 * Marks the transaction so user edits can map the range without remapping this replacement.
 */
function writeTranscript(
  editor: Editor,
  range: VoiceRange,
  final: string,
  interim: string
): VoiceRange {
  const content = transcriptContent(final, interim)
  const to =
    range.from +
    content.reduce((length, node) => length + (node.text?.length ?? 0), 0)
  editor
    .chain()
    .command(({ tr }) => {
      tr.setMeta(VOICE_TRANSACTION, true)
      return true
    })
    .insertContentAt(range, content)
    .setTextSelection(to)
    .run()
  return { from: range.from, to }
}

/** Removes interim styling from an existing span and focuses its end; cleared documents are ignored. */
function settleTranscript(
  editor: Editor | null,
  range: VoiceRange | null
): void {
  if (
    !editor ||
    !range ||
    range.to <= range.from ||
    range.to > editor.state.doc.content.size
  )
    return
  editor
    .chain()
    .setTextSelection(range)
    .unsetMark(InterimTranscript.name)
    .setTextSelection(range.to)
    .focus()
    .run()
}

/**
 * Owns dictation's document range, recognition updates and keyboard shortcut for one composer.
 * Returns microphone state and controls; user transactions map the range, and send releases it before clearing.
 */
export function useComposerDictation(
  editorRef: React.RefObject<Editor | null>,
  disabled: boolean
) {
  const voice = useSpeechRecognition()
  const { transcript, listening, supported, start, stop } = voice
  const rangeRef = React.useRef<VoiceRange | null>(null)
  const wasListeningRef = React.useRef(false)

  /** Anchors a new dictation at the current selection, separating it from a preceding word once. */
  const begin = React.useCallback(() => {
    const editor = editorRef.current
    if (!editor || disabled) return
    editor.commands.focus()
    const at = editor.state.selection.to
    const $at = editor.state.doc.resolve(at)
    const previous =
      $at.parentOffset > 0
        ? $at.parent.textBetween($at.parentOffset - 1, $at.parentOffset)
        : ""
    if (previous && !/\s/.test(previous))
      editor.commands.insertContentAt(at, { type: "text", text: " " })
    const anchor = editor.state.selection.to
    rangeRef.current = { from: anchor, to: anchor }
    start()
  }, [editorRef, disabled, start])

  /** Releases the range before sending/clearing the document; recognition is stopped separately. */
  const releaseRange = React.useCallback(() => {
    rangeRef.current = null
  }, [])

  /** Keeps the spoken span aligned through user edits; voice-owned replacements update their own range. */
  const mapTransaction = React.useCallback((transaction: Transaction) => {
    const range = rangeRef.current
    if (
      !range ||
      !transaction.docChanged ||
      transaction.getMeta(VOICE_TRANSACTION)
    )
      return
    rangeRef.current = {
      from: transaction.mapping.map(range.from),
      to: transaction.mapping.map(range.to),
    }
  }, [])

  React.useEffect(() => {
    const editor = editorRef.current
    const range = rangeRef.current
    if (!editor || !range || (!transcript.final && !transcript.interim)) return
    rangeRef.current = writeTranscript(
      editor,
      range,
      transcript.final,
      transcript.interim
    )
  }, [editorRef, transcript])
  React.useEffect(() => {
    const stopped = wasListeningRef.current && !listening
    wasListeningRef.current = listening
    if (!stopped) return
    const range = rangeRef.current
    rangeRef.current = null
    settleTranscript(editorRef.current, range)
  }, [editorRef, listening])
  React.useEffect(() => {
    if (!supported || disabled) return
    /** Toggles recognition for Ctrl/Cmd+Shift+Space without claiming other browser shortcuts. */
    const handleShortcut = (event: KeyboardEvent) => {
      if (
        event.code !== "Space" ||
        !event.shiftKey ||
        (!event.metaKey && !event.ctrlKey)
      )
        return
      event.preventDefault()
      if (listening) stop()
      else begin()
    }
    document.addEventListener("keydown", handleShortcut)
    return () => document.removeEventListener("keydown", handleShortcut)
  }, [supported, disabled, listening, stop, begin])
  return {
    supported,
    listening,
    onDevice: voice.onDevice,
    error: voice.error,
    begin,
    stop,
    releaseRange,
    mapTransaction,
  }
}
