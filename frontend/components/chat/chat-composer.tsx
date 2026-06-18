"use client"

import * as React from "react"
import { ArrowUp } from "lucide-react"
import { EditorContent, useEditor } from "@tiptap/react"
import { Extension } from "@tiptap/core"
import StarterKit from "@tiptap/starter-kit"
import Placeholder from "@tiptap/extension-placeholder"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { docToMarkdown } from "@/lib/tiptap-markdown"
import { FluxThrobber } from "./flux-throbber"
import { ModelSelector } from "./model-selector"

// Tab / Shift+Tab indent / outdent list items. Enter and Shift+Enter are
// handled in editorProps.handleKeyDown so they run before StarterKit's own
// keymaps (otherwise hardBreak's Shift+Enter wins and inserts a <br> instead
// of splitting into a new block, which stops `- ` from starting a list).
const ComposerKeys = Extension.create({
  name: "composerKeys",
  priority: 1000,
  addKeyboardShortcuts() {
    return {
      Tab: () => this.editor.commands.sinkListItem("listItem") || true,
      "Shift-Tab": () => this.editor.commands.liftListItem("listItem") || true,
    }
  },
})

export function ChatComposer({
  onSend,
  disabled = false,
  placeholder = "Talk to Gork...",
}: {
  onSend: (content: string) => Promise<void> | void
  disabled?: boolean
  placeholder?: string
}) {
  const [model, setModel] = React.useState("")
  const [isEmpty, setIsEmpty] = React.useState(true)

  const onSendRef = React.useRef(onSend)
  onSendRef.current = onSend
  const submitRef = React.useRef<() => void>(() => {})
  const editorRef = React.useRef<ReturnType<typeof useEditor>>(null)

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
    ],
    editorProps: {
      attributes: {
        class:
          "tiptap max-h-72 min-h-20 overflow-y-auto px-3.5 py-3 text-sm leading-relaxed",
      },
      handleKeyDown: (_view, event) => {
        if (event.key !== "Enter") return false
        const ed = editorRef.current
        if (!ed) return false

        if (event.shiftKey) {
          // New line without sending: continue the list, add a code newline, or
          // split into a fresh paragraph so `- ` / `1.` can start a list.
          if (ed.isActive("codeBlock")) ed.commands.newlineInCode()
          else if (ed.can().splitListItem("listItem"))
            ed.commands.splitListItem("listItem")
          else ed.commands.splitBlock()
          event.preventDefault()
          return true
        }

        // Plain Enter sends, except inside a code block (where it adds a line).
        if (ed.isActive("codeBlock")) return false
        event.preventDefault()
        submitRef.current()
        return true
      },
    },
    onCreate: ({ editor }) => setIsEmpty(editor.isEmpty),
    onUpdate: ({ editor }) => setIsEmpty(editor.isEmpty),
  })
  editorRef.current = editor

  const submit = React.useCallback(() => {
    if (!editor || editor.isEmpty || disabled) return
    const markdown = docToMarkdown(editor.getJSON())
    if (!markdown.trim()) return
    editor.commands.clearContent()
    setIsEmpty(true)
    onSendRef.current(markdown)
  }, [editor, disabled])
  submitRef.current = submit

  React.useEffect(() => {
    editor?.setEditable(!disabled)
  }, [editor, disabled])

  const canSend = !isEmpty && !disabled

  return (
    <div
      className={cn(
        "rounded-2xl border border-input shadow-sm transition-colors",
        "focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50",
        disabled && "opacity-70"
      )}
    >
      <EditorContent editor={editor} />
      <div className="flex items-center gap-2 px-2.5 pb-2">
        <ModelSelector value={model} onValueChange={setModel} />
        <Button
          aria-label="Send"
          size="icon-sm"
          className="ml-auto rounded-full"
          disabled={!canSend}
          onClick={submit}
        >
          {disabled ? (
            <FluxThrobber
              mode="thinking"
              size={18}
              className="rounded-full bg-transparent"
            />
          ) : (
            <ArrowUp />
          )}
        </Button>
      </div>
    </div>
  )
}
