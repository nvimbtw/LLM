"use client"

import * as React from "react"
import { ArrowUp, Square } from "lucide-react"
import { EditorContent, useEditor } from "@tiptap/react"
import { Extension } from "@tiptap/core"
import StarterKit from "@tiptap/starter-kit"
import Placeholder from "@tiptap/extension-placeholder"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { docToMarkdown } from "@/lib/tiptap-markdown"
import { matchStandaloneCommand, runSlashCommand } from "@/lib/commands"
import { useSettings } from "@/components/settings/settings-provider"
import { ModelSelector } from "./model-selector"
import { BudgetSelector, type ThinkBudget } from "./budget-selector"

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
  onCancel,
  placeholder = "Message Slate…",
}: {
  onSend: (
    content: string,
    modelId: string,
    budget: ThinkBudget
  ) => Promise<void> | void
  disabled?: boolean
  // Stops the turn in flight. A reply is a queued job now, so it can be waiting
  // behind a long run with nothing to watch — being able to take it back out of
  // the line matters more than it did when a turn started instantly.
  onCancel?: () => void
  placeholder?: string
}) {
  const [model, setModel] = React.useState("")
  // Per-message, and deliberately not a stored setting: how hard you want the
  // model to think is a property of the question you are about to ask, not a
  // preference. It does persist for the life of the composer, so a run of
  // similar questions is not a run of re-picking.
  const [budget, setBudget] = React.useState<ThinkBudget>("auto")
  const [isEmpty, setIsEmpty] = React.useState(true)
  // True when the current content is a standalone, recognised slash command —
  // drives the blue highlight and the "run as command instead of send" path.
  const [isCommand, setIsCommand] = React.useState(false)
  const { settings, setSetting, setCommandOpen } = useSettings()

  // Three latest-value boxes for Tiptap's `handleKeyDown`, which is configured
  // once and would otherwise hold the first render's closures forever. They are
  // written from effects, not during render: a render can be thrown away or
  // replayed, and a ref written during one is a mutation that escapes it.
  // Reading them post-commit is safe here because `handleKeyDown` only ever runs
  // from a real keystroke.
  const onSendRef = React.useRef(onSend)
  const submitRef = React.useRef<() => void>(() => {})
  const editorRef = React.useRef<ReturnType<typeof useEditor>>(null)

  React.useEffect(() => {
    onSendRef.current = onSend
  })

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
    onUpdate: ({ editor }) => {
      setIsEmpty(editor.isEmpty)
      setIsCommand(matchStandaloneCommand(editor.getText()) !== null)
    },
  })

  React.useEffect(() => {
    editorRef.current = editor
  })

  const submit = React.useCallback(() => {
    if (!editor || editor.isEmpty || disabled) return

    // A standalone, recognised slash command runs locally and is never sent to
    // the model. Anything else (including a "/" mid-sentence) is an ordinary
    // message.
    const text = editor.getText()
    if (matchStandaloneCommand(text)) {
      runSlashCommand(text, {
        settings,
        setSetting,
        openSettings: () => setCommandOpen(true),
      })
      editor.commands.clearContent()
      setIsEmpty(true)
      setIsCommand(false)
      return
    }

    const markdown = docToMarkdown(editor.getJSON())
    if (!markdown.trim()) return
    editor.commands.clearContent()
    setIsEmpty(true)
    onSendRef.current(markdown, model, budget)
  }, [editor, disabled, model, budget, settings, setSetting, setCommandOpen])

  React.useEffect(() => {
    submitRef.current = submit
  })

  React.useEffect(() => {
    editor?.setEditable(!disabled)
  }, [editor, disabled])

  const canSend = !isEmpty && !disabled

  return (
    <div
      className={cn(
        "rounded-[0.875rem] border border-input shadow-sm transition-colors",
        "focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50",
        disabled && "opacity-70"
      )}
    >
      {/* A recognised standalone command colours the whole line blue (primary),
          signalling it'll run as a command rather than be sent to the model. */}
      <div
        className={cn(
          isCommand &&
            "[&_.tiptap]:font-medium [&_.tiptap]:text-primary [&_.tiptap]:caret-primary"
        )}
      >
        <EditorContent editor={editor} />
      </div>
      <div className="flex items-center gap-2 px-2.5 pb-2">
        <ModelSelector value={model} onValueChange={setModel} />
        <BudgetSelector value={budget} onValueChange={setBudget} />
        <Button
          aria-label={disabled && onCancel ? "Stop" : "Send"}
          size="icon-sm"
          className="ml-auto rounded-full"
          // While a turn is in flight the button becomes the way to stop it, so
          // it stays live rather than going inert with a mark spinning on it.
          disabled={disabled ? !onCancel : !canSend}
          onClick={disabled ? onCancel : submit}
        >
          {/* A stop glyph rather than a mark of activity: while a turn is in
              flight this button *is* the way to stop it, and its own label says so.
              What the model is doing is said once, by the throbber in the thread —
              it does not also need saying at 32px inside a control. */}
          {disabled ? <Square className="fill-current" /> : <ArrowUp />}
        </Button>
      </div>
    </div>
  )
}
