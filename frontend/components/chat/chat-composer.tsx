"use client"

import * as React from "react"
import { ArrowUp, Square } from "lucide-react"
import { EditorContent, useEditor } from "@tiptap/react"
import { Extension } from "@tiptap/core"
import StarterKit from "@tiptap/starter-kit"
import Placeholder from "@tiptap/extension-placeholder"
import { useTheme } from "next-themes"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { docToMarkdown } from "@/lib/tiptap-markdown"
import {
  findCommand,
  getCommandSuggestions,
  matchStandaloneCommand,
  runSlashCommand,
  type CommandContext,
  type SlashCommandDef,
} from "@/lib/commands"
import { useSettings } from "@/components/settings/settings-provider"
import { useModels } from "@/hooks/use-models"
import { ModelSelector } from "./model-selector"
import { BudgetSelector, type ThinkBudget } from "./budget-selector"
import { SlashCommandMenu } from "./slash-command-menu"

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
  const [isCommand, setIsCommand] = React.useState(false)
  const [editorText, setEditorText] = React.useState("")
  const [selectedIndex, setSelectedIndex] = React.useState(0)
  const [menuDismissed, setMenuDismissed] = React.useState(false)

  const { settings, setSetting, setCommandOpen } = useSettings()
  const { models } = useModels()
  const { theme, setTheme } = useTheme()

  const onSendRef = React.useRef(onSend)
  const submitRef = React.useRef<() => void>(() => {})
  const editorRef = React.useRef<ReturnType<typeof useEditor>>(null)

  React.useEffect(() => {
    onSendRef.current = onSend
  })

  const clearComposer = React.useCallback(() => {
    setIsEmpty(true)
    setIsCommand(false)
    setEditorText("")
    setMenuDismissed(false)
    setSelectedIndex(0)
  }, [])

  const commandContext: CommandContext = React.useMemo(
    () => ({
      settings,
      setSetting,
      openSettings: () => setCommandOpen(true),
      model,
      setModel,
      models,
      budget,
      setBudget,
      theme,
      setTheme,
      clearComposer,
    }),
    [settings, setSetting, setCommandOpen, model, models, budget, theme, setTheme, clearComposer]
  )

  const commandContextRef = React.useRef(commandContext)
  React.useEffect(() => {
    commandContextRef.current = commandContext
  })

  const [prevEditorText, setPrevEditorText] = React.useState("")
  if (editorText !== prevEditorText) {
    setPrevEditorText(editorText)
    setSelectedIndex(0)
  }

  const isSlashActive =
    editorText.trimStart().startsWith("/") &&
    !editorText.includes("\n") &&
    !menuDismissed

  const suggestionsResult = React.useMemo(() => {
    if (!isSlashActive) return null
    return getCommandSuggestions(editorText, commandContext)
  }, [isSlashActive, editorText, commandContext])

  const suggestionsCount = React.useMemo(() => {
    if (!suggestionsResult) return 0
    if (suggestionsResult.mode === "command") {
      return suggestionsResult.suggestions.length
    }
    const { data } = suggestionsResult
    if (data.type === "model") return data.options.length
    if (data.type === "budget") return data.options.length
    if (data.type === "toggle") return data.options.length
    if (data.type === "theme") return data.options.length
    if (data.type === "number") return data.info.presets.length
    if (data.type === "text") return data.info.templates.length
    return 1
  }, [suggestionsResult])

  const handleSelectCommand = React.useCallback(
    (cmd: SlashCommandDef) => {
      const ed = editorRef.current
      if (!ed) return

      if (cmd.kind === "action") {
        runSlashCommand(`/${cmd.name}`, commandContextRef.current)
        ed.commands.clearContent()
        setIsEmpty(true)
        setIsCommand(false)
        setEditorText("")
        setMenuDismissed(false)
        return
      }

      ed.commands.setContent(`/${cmd.name} `)
      ed.commands.focus("end")
      setEditorText(`/${cmd.name} `)
      setSelectedIndex(0)
    },
    []
  )

  const handleSelectOption = React.useCallback(
    (optionValue: string, executeImmediately = true) => {
      const ed = editorRef.current
      if (!ed) return

      const text = ed.getText().trim()
      const commandWord = text.slice(1).split(/\s+/)[0]
      const fullCmd = `/${commandWord} ${optionValue}`.trim()

      if (executeImmediately) {
        runSlashCommand(fullCmd, commandContextRef.current)
        ed.commands.clearContent()
        setIsEmpty(true)
        setIsCommand(false)
        setEditorText("")
        setMenuDismissed(false)
        return
      }

      ed.commands.setContent(`${fullCmd} `)
      ed.commands.focus("end")
      setEditorText(`${fullCmd} `)
    },
    []
  )

  const selectCurrentSuggestion = React.useCallback(() => {
    if (!suggestionsResult || suggestionsCount === 0) return false
    const ed = editorRef.current
    if (!ed) return false

    if (suggestionsResult.mode === "command") {
      const target = suggestionsResult.suggestions[selectedIndex]?.command
      if (target) {
        handleSelectCommand(target)
        return true
      }
    } else if (suggestionsResult.mode === "argument") {
      const { command, data } = suggestionsResult
      if (data.type === "model" && data.options[selectedIndex]) {
        handleSelectOption(data.options[selectedIndex].model.id, true)
        return true
      }
      if (data.type === "budget" && data.options[selectedIndex]) {
        handleSelectOption(data.options[selectedIndex].value, true)
        return true
      }
      if (data.type === "toggle" && data.options[selectedIndex]) {
        handleSelectOption(data.options[selectedIndex].value, true)
        return true
      }
      if (data.type === "theme" && data.options[selectedIndex]) {
        handleSelectOption(data.options[selectedIndex].value, true)
        return true
      }
      if (data.type === "number" && data.info.presets[selectedIndex]) {
        const val = data.info.presets[selectedIndex].value.toFixed(
          data.info.precision
        )
        handleSelectOption(val, true)
        return true
      }
      if (data.type === "text" && data.info.templates[selectedIndex]) {
        handleSelectOption(data.info.templates[selectedIndex].value, true)
        return true
      }
      if (data.type === "action") {
        runSlashCommand(`/${command.name}`, commandContextRef.current)
        ed.commands.clearContent()
        setIsEmpty(true)
        setIsCommand(false)
        setEditorText("")
        setMenuDismissed(false)
        return true
      }
    }
    return false
  }, [
    suggestionsResult,
    suggestionsCount,
    selectedIndex,
    handleSelectCommand,
    handleSelectOption,
  ])

  const selectCurrentSuggestionRef = React.useRef(selectCurrentSuggestion)
  React.useEffect(() => {
    selectCurrentSuggestionRef.current = selectCurrentSuggestion
  })

  const suggestionsCountRef = React.useRef(suggestionsCount)
  React.useEffect(() => {
    suggestionsCountRef.current = suggestionsCount
  })

  const isSlashActiveRef = React.useRef(isSlashActive)
  React.useEffect(() => {
    isSlashActiveRef.current = isSlashActive
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
        // Handle fuzzy menu navigation when active
        if (isSlashActiveRef.current && suggestionsCountRef.current > 0) {
          if (event.key === "ArrowDown") {
            event.preventDefault()
            setSelectedIndex((prev) => (prev + 1) % suggestionsCountRef.current)
            return true
          }
          if (event.key === "ArrowUp") {
            event.preventDefault()
            setSelectedIndex(
              (prev) =>
                (prev - 1 + suggestionsCountRef.current) %
                suggestionsCountRef.current
            )
            return true
          }
          if (event.key === "Tab") {
            event.preventDefault()
            if (selectCurrentSuggestionRef.current()) return true
          }
          if (event.key === "Escape") {
            event.preventDefault()
            setMenuDismissed(true)
            return true
          }
        }

        // Escape with no menu to dismiss: give up focus, which drops the
        // composer's highlighted ring — the same "escape gets you out"
        // behaviour as the menu case above, one level up.
        if (event.key === "Escape") {
          event.preventDefault()
          editorRef.current?.commands.blur()
          return true
        }

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

        // If in command discovery mode and user hits Enter on a highlighted item
        if (
          isSlashActiveRef.current &&
          suggestionsCountRef.current > 0 &&
          !ed.getText().includes(" ")
        ) {
          event.preventDefault()
          if (selectCurrentSuggestionRef.current()) return true
        }

        event.preventDefault()
        submitRef.current()
        return true
      },
    },
    onCreate: ({ editor }) => {
      setIsEmpty(editor.isEmpty)
      const text = editor.getText()
      setEditorText(text)
      setIsCommand(matchStandaloneCommand(text) !== null)
    },
    onUpdate: ({ editor }) => {
      const text = editor.getText()
      setIsEmpty(editor.isEmpty)
      setEditorText(text)
      setMenuDismissed(false)
      const word = text.trim().slice(1).split(/\s+/)[0]
      setIsCommand(
        matchStandaloneCommand(text) !== null ||
          (text.trim().startsWith("/") && Boolean(findCommand(word)))
      )
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
      runSlashCommand(text, commandContextRef.current)
      editor.commands.clearContent()
      setIsEmpty(true)
      setIsCommand(false)
      setEditorText("")
      setMenuDismissed(false)
      return
    }

    const markdown = docToMarkdown(editor.getJSON())
    if (!markdown.trim()) return
    editor.commands.clearContent()
    setIsEmpty(true)
    setEditorText("")
    onSendRef.current(markdown, model, budget)
  }, [editor, disabled, model, budget])

  React.useEffect(() => {
    submitRef.current = submit
  })

  React.useEffect(() => {
    editor?.setEditable(!disabled)
  }, [editor, disabled])

  // Type anywhere on the page and it lands in the composer — the same
  // "just start typing" feel as any chat app. A keystroke that isn't already
  // headed into some other editable field (an input, a textarea, another
  // contenteditable) is redirected here instead of being dropped on the
  // floor. Escape reverses it by blurring (handled in editorProps above,
  // since at that point the editor already has focus and owns the event).
  React.useEffect(() => {
    if (disabled) return

    const handleGlobalKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return
      // Not a single printable character — arrows, Tab, Backspace, function
      // keys, etc. all have multi-character `key` values and fall through.
      if (event.key.length !== 1) return

      const target = event.target as HTMLElement | null
      const tag = target?.tagName
      const alreadyEditable =
        target?.isContentEditable || tag === "INPUT" || tag === "TEXTAREA"
      if (alreadyEditable) return

      const ed = editorRef.current
      if (!ed) return

      event.preventDefault()
      ed.chain().focus("end").insertContent(event.key).run()
    }

    document.addEventListener("keydown", handleGlobalKeyDown)
    return () => document.removeEventListener("keydown", handleGlobalKeyDown)
  }, [disabled])

  const canSend = !isEmpty && !disabled

  return (
    <div
      className={cn(
        "relative rounded-[0.875rem] border border-input shadow-sm transition-colors",
        "focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50",
        disabled && "opacity-70"
      )}
    >
      {/* Floating Slash Command Fuzzy Finder — always mounted so it can play
          its own exit animation; it renders nothing once `open` has been
          false long enough to finish that. */}
      <SlashCommandMenu
        open={isSlashActive}
        input={editorText}
        selectedIndex={selectedIndex}
        onSelectCommand={handleSelectCommand}
        onSelectOption={handleSelectOption}
        ctx={commandContext}
      />

      {/* A recognised command colours the line blue (primary),
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

