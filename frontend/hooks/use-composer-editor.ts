"use client"

import * as React from "react"
import { useEditor } from "@tiptap/react"
import type { Editor } from "@tiptap/core"
import type { Transaction } from "@tiptap/pm/state"
import StarterKit from "@tiptap/starter-kit"
import Placeholder from "@tiptap/extension-placeholder"

import { ComposerKeys, InterimTranscript } from "@/lib/composer-editor"
import { findCommand, matchStandaloneCommand } from "@/lib/commands"
import type { useComposerCommands } from "./use-composer-commands"

type CommandMenu = ReturnType<typeof useComposerCommands>

export interface ComposerEditorSnapshot {
  isEmpty: boolean
  isCommand: boolean
  text: string
}

export interface ComposerEditorOptions {
  editorRef: React.RefObject<Editor | null>
  submitRef: React.RefObject<() => void>
  commandMenuRef: React.RefObject<CommandMenu>
  onChange: (snapshot: ComposerEditorSnapshot) => void
  mapTransaction: (transaction: Transaction) => void
  placeholder: string
  disabled: boolean
  draft?: { text: string }
}

/**
 * Handles the menu's bottom-up arrow order, completion and dismissal, returning whether the key was consumed.
 * Index zero remains the best match nearest the editor rather than reversing the suggestion data.
 */
function handleMenuKey(menu: CommandMenu, event: KeyboardEvent): boolean {
  if (!menu.isSlashActive) return false
  if (event.key === "Escape") {
    event.preventDefault()
    menu.setMenuDismissed(true)
    return true
  }
  if (menu.suggestionsCount === 0) return false
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault()
    const direction = event.key === "ArrowDown" ? -1 : 1
    menu.setSelectedIndex(
      (previous) =>
        (previous + direction + menu.suggestionsCount) % menu.suggestionsCount
    )
    return true
  }
  if (event.key !== "Tab") return false
  event.preventDefault()
  return menu.selectCurrentSuggestion()
}

/** Resolves menu keys before editor keymaps, preserving code newlines, list splitting and Enter submission. */
function handleEditorKey(
  editor: Editor | null,
  menu: CommandMenu,
  submit: () => void,
  event: KeyboardEvent
): boolean {
  if (handleMenuKey(menu, event)) return true
  if (event.key === "Escape") {
    event.preventDefault()
    editor?.commands.blur()
    return true
  }
  if (event.key !== "Enter" || !editor) return false
  if (event.shiftKey) {
    if (editor.isActive("codeBlock")) editor.commands.newlineInCode()
    else if (editor.can().splitListItem("listItem"))
      editor.commands.splitListItem("listItem")
    else editor.commands.splitBlock()
    event.preventDefault()
    return true
  }
  if (editor.isActive("codeBlock")) return false
  event.preventDefault()
  if (
    menu.isSlashActive &&
    menu.suggestionsCount > 0 &&
    !editor.getText().includes(" ") &&
    menu.selectCurrentSuggestion()
  )
    return true
  submit()
  return true
}

/** Reads the plain text and local command status for rendering and draft-token scheduling. */
function editorSnapshot(editor: Editor): ComposerEditorSnapshot {
  const text = editor.getText()
  const word = text.trim().slice(1).split(/\s+/)[0]
  return {
    isEmpty: editor.isEmpty,
    text,
    isCommand:
      matchStandaloneCommand(text) !== null ||
      (text.trim().startsWith("/") && Boolean(findCommand(word))),
  }
}

/**
 * Owns Tiptap setup, input routing, editability and literal draft loading for a composer.
 * Draft identity triggers reloads; plain-text paragraphs keep HTML-looking questions intact.
 */
export function useComposerEditor(options: ComposerEditorOptions) {
  const {
    editorRef,
    submitRef,
    commandMenuRef,
    onChange,
    mapTransaction,
    placeholder,
    disabled,
    draft,
  } = options
  const editor = useEditor({
    immediatelyRender: false,
    extensions: [
      StarterKit.configure({
        heading: false,
        blockquote: false,
        strike: false,
        horizontalRule: false,
        link: false,
      }),
      Placeholder.configure({ placeholder }),
      ComposerKeys,
      InterimTranscript,
    ],
    editorProps: {
      attributes: {
        class:
          "tiptap max-h-72 overflow-y-auto px-3.5 py-3 text-sm leading-relaxed",
        role: "textbox",
        "aria-label": placeholder,
        "aria-multiline": "true",
      },
      handleKeyDown: (_view, event) =>
        handleEditorKey(
          editorRef.current,
          commandMenuRef.current,
          submitRef.current,
          event
        ),
    },
    onCreate: ({ editor }) => onChange(editorSnapshot(editor)),
    onUpdate: ({ editor }) => {
      commandMenuRef.current.setMenuDismissed(false)
      onChange(editorSnapshot(editor))
    },
    onTransaction: ({ transaction }) => mapTransaction(transaction),
  })
  React.useEffect(() => {
    editorRef.current = editor
  }, [editor, editorRef])
  React.useEffect(() => {
    editor?.setEditable(!disabled)
  }, [editor, disabled])
  React.useEffect(() => {
    const current = editorRef.current
    if (!current || !draft) return
    current.commands.setContent({
      type: "doc",
      content: draft.text.split("\n").map((line) => ({
        type: "paragraph",
        ...(line ? { content: [{ type: "text", text: line }] } : {}),
      })),
    })
    current.commands.focus("end")
  }, [editorRef, draft])
  React.useEffect(() => {
    if (disabled) return
    /** Routes printable page keystrokes into the composer while respecting other editable fields and controls. */
    const handleGlobalKeyDown = (event: KeyboardEvent) => {
      if (
        event.ctrlKey ||
        event.metaKey ||
        event.altKey ||
        event.key.length !== 1
      )
        return
      const target = event.target instanceof HTMLElement ? event.target : null
      if (
        target?.isContentEditable ||
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.closest("button, [role='button']")
      )
        return
      const current = editorRef.current
      if (!current) return
      event.preventDefault()
      current.chain().focus("end").insertContent(event.key).run()
    }
    document.addEventListener("keydown", handleGlobalKeyDown)
    return () => document.removeEventListener("keydown", handleGlobalKeyDown)
  }, [editorRef, disabled])
  return editor
}
