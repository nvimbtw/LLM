"use client"

import * as React from "react"
import { ArrowUp, Paperclip, Square } from "lucide-react"
import { EditorContent, useEditor } from "@tiptap/react"
import { Extension, type Editor } from "@tiptap/core"
import type { ResolvedPos } from "@tiptap/pm/model"
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
import { AttachMenu } from "./attach-menu"
import { AttachmentStrip, type PendingAttachment } from "./attachment-strip"
import { ComposerFooter } from "./composer-footer"

// One level of indentation inside a code block. Two spaces, matching the
// nesting width `docToMarkdown` uses for lists, so everything the composer
// indents is indented by the same amount.
const CODE_INDENT = "  "

// Tab inside a code block indents, rather than doing nothing at all.
// `sinkListItem` fails outside a list and the `|| true` below swallows the key
// to keep Tab from tabbing focus out of the composer — so without this branch a
// fenced block is the one place where Tab is eaten and you have to type spaces
// by hand.
//
// Whole lines, not the cursor: a Tab with something selected indents every line
// the selection touches, which is what the key means in an editor and the only
// way to shift a block of code that's already written.
function indentCode(editor: Editor, outdent: boolean): boolean {
  const { state } = editor
  const { from, to, empty } = state.selection

  // Which side of the selection actually sits in the block matters: dragging
  // upward from the last line lands the *start* of the range on the position
  // just before the block, where the enclosing node is the document and every
  // offset below would be measured against the wrong text.
  const $from = state.doc.resolve(from)
  const $to = state.doc.resolve(to)
  const $in = codeBlockDepth($from) !== null ? $from : $to
  const depth = codeBlockDepth($in)
  if (depth === null) return false

  const start = $in.start(depth)
  const end = $in.end(depth)
  const text = $in.node(depth).textContent

  // Nothing selected: Tab is an insert at the cursor, like any editor.
  if (empty && !outdent) {
    editor.view.dispatch(state.tr.insertText(CODE_INDENT, from))
    return true
  }

  // Clamped to the block, so a selection running past either edge indents the
  // code it covers rather than reaching for lines that aren't there.
  const selFrom = Math.max(from, start)
  const selTo = Math.min(to, end)

  // The document position of the first character of every line the selection
  // touches, collected up front and applied back-to-front — an edit only ever
  // shifts positions after it, so working downward leaves the ones still to do
  // untouched and no position mapping is needed.
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
    // Take back one indent's worth: a tab, or up to CODE_INDENT spaces. Lines
    // shallower than a full indent lose whatever they have rather than nothing.
    const lead = text.slice(pos - start).match(/^(\t| {1,2})/)
    if (lead) tr.delete(pos, pos + lead[1].length)
  }

  if (tr.docChanged) editor.view.dispatch(tr)
  return true
}

// The depth at which this position sits inside a code block, or null if it
// isn't in one. `editor.isActive("codeBlock")` is the usual test and the wrong
// one here — it answers for the selection as a whole, including ranges that
// merely touch the block from outside.
function codeBlockDepth($pos: ResolvedPos): number | null {
  for (let d = $pos.depth; d > 0; d--) {
    if ($pos.node(d).type.name === "codeBlock") return d
  }
  return null
}

// Tab / Shift+Tab indent / outdent list items, or code inside a fenced block.
// Enter and Shift+Enter are handled in editorProps.handleKeyDown so they run
// before StarterKit's own keymaps (otherwise hardBreak's Shift+Enter wins and
// inserts a <br> instead of splitting into a new block, which stops `- ` from
// starting a list).
const ComposerKeys = Extension.create({
  name: "composerKeys",
  priority: 1000,
  addKeyboardShortcuts() {
    return {
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

export function ChatComposer({
  onSend,
  disabled = false,
  onCancel,
  placeholder = "Message Slate…",
  title,
  onRenameTitle,
  contextTokensUsed = 0,
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
  // The footer below the composer — see composer-footer.tsx. Optional
  // so the composer still renders (minus that row) wherever a caller hasn't
  // got a conversation to hang it off yet.
  title?: string
  onRenameTitle?: (title: string) => void
  contextTokensUsed?: number
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
  // Staged, not yet wired to sending — see attachment-strip.tsx. Dropped on
  // submit for now (`clearAttachments` in `submit`, below).
  const [attachments, setAttachments] = React.useState<PendingAttachment[]>([])
  const [isDraggingFiles, setIsDraggingFiles] = React.useState(false)

  const { settings, setSetting, setCommandOpen } = useSettings()
  const { models } = useModels()
  const { theme, setTheme } = useTheme()

  const onSendRef = React.useRef(onSend)
  const submitRef = React.useRef<() => void>(() => {})
  const editorRef = React.useRef<ReturnType<typeof useEditor>>(null)

  React.useEffect(() => {
    onSendRef.current = onSend
  })

  // Object URLs (image thumbnails) are only ever revoked here and in
  // `removeAttachment` below — kept as the single place that owns their
  // lifetime so none leak past a re-render or an unmount.
  const clearAttachments = React.useCallback(() => {
    setAttachments((prev) => {
      for (const a of prev) if (a.previewUrl) URL.revokeObjectURL(a.previewUrl)
      return []
    })
  }, [])

  const addFiles = React.useCallback((files: FileList | File[]) => {
    const incoming = Array.from(files)
    if (incoming.length === 0) return
    setAttachments((prev) => [
      ...prev,
      ...incoming.map((file): PendingAttachment => {
        const isImage = file.type.startsWith("image/")
        return {
          id: crypto.randomUUID(),
          file,
          kind: isImage ? "image" : "file",
          previewUrl: isImage ? URL.createObjectURL(file) : undefined,
        }
      }),
    ])
  }, [])

  const removeAttachment = React.useCallback((id: string) => {
    setAttachments((prev) => {
      const target = prev.find((a) => a.id === id)
      if (target?.previewUrl) URL.revokeObjectURL(target.previewUrl)
      return prev.filter((a) => a.id !== id)
    })
  }, [])

  // Revoke whatever's left on unmount. `attachments` itself isn't a
  // dependency — this should only run once, reading whatever the ref holds
  // at that moment — so it's mirrored into a ref rather than closed over.
  const attachmentsRef = React.useRef(attachments)
  React.useEffect(() => {
    attachmentsRef.current = attachments
  })
  React.useEffect(() => {
    return () => {
      for (const a of attachmentsRef.current) {
        if (a.previewUrl) URL.revokeObjectURL(a.previewUrl)
      }
    }
  }, [])

  const clearComposer = React.useCallback(() => {
    setIsEmpty(true)
    setIsCommand(false)
    setEditorText("")
    setMenuDismissed(false)
    setSelectedIndex(0)
    clearAttachments()
  }, [clearAttachments])

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
    [
      settings,
      setSetting,
      setCommandOpen,
      model,
      models,
      budget,
      theme,
      setTheme,
      clearComposer,
    ]
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

  const handleSelectCommand = React.useCallback((cmd: SlashCommandDef) => {
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
  }, [])

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
          // Inverted against the index, because the menu is painted bottom-up:
          // index 0 is the best match and it sits at the *bottom* of the list,
          // nearest the line you are typing on. So walking the highlight
          // downward walks toward index 0. Flipping it here rather than
          // reversing the data keeps `selectedIndex` meaning "nth best"
          // everywhere else — Tab autocomplete and Enter both read it.
          if (event.key === "ArrowDown") {
            event.preventDefault()
            setSelectedIndex(
              (prev) =>
                (prev - 1 + suggestionsCountRef.current) %
                suggestionsCountRef.current
            )
            return true
          }
          if (event.key === "ArrowUp") {
            event.preventDefault()
            setSelectedIndex((prev) => (prev + 1) % suggestionsCountRef.current)
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
    // Attachments aren't wired to the send path yet — they're discarded
    // here rather than passed through. See attachment-strip.tsx.
    clearAttachments()
    onSendRef.current(markdown, model, budget)
  }, [editor, disabled, model, budget, clearAttachments])

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

  const contextWindow =
    models.find((m) => m.id === model)?.contextWindow ?? null

  // A file drag carries "Files" in dataTransfer.types from the moment it
  // enters the window — checking for that (rather than inspecting the files
  // themselves, which isn't allowed until drop) is what keeps this from
  // reacting to ordinary in-page drags, like selecting and dragging text.
  const isFileDrag = (e: React.DragEvent) =>
    !disabled && Boolean(e.dataTransfer?.types.includes("Files"))

  // dragenter/dragleave fire on every child as the pointer crosses it, not
  // just once for the composer as a whole — a depth counter is the standard
  // fix, since only the transition from/to zero should flip the overlay.
  const dragDepthRef = React.useRef(0)

  const handleDragEnter = (e: React.DragEvent) => {
    if (!isFileDrag(e)) return
    e.preventDefault()
    dragDepthRef.current += 1
    setIsDraggingFiles(true)
  }
  const handleDragOver = (e: React.DragEvent) => {
    if (!isFileDrag(e)) return
    e.preventDefault()
  }
  const handleDragLeave = (e: React.DragEvent) => {
    if (!isFileDrag(e)) return
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1)
    if (dragDepthRef.current === 0) setIsDraggingFiles(false)
  }
  const handleDrop = (e: React.DragEvent) => {
    if (!isFileDrag(e)) return
    e.preventDefault()
    dragDepthRef.current = 0
    setIsDraggingFiles(false)
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files)
  }

  return (
    // The composer is its own card again — border/radius/shadow/focus-ring
    // all its own — and the footer (see composer-footer.tsx) sits below it
    // as a separate, quieter element, not sharing the card's outline.
    <div className="flex flex-col gap-1.5">
      <div
        className={cn(
          "relative overflow-hidden rounded-[0.875rem] border border-input shadow-sm transition-colors",
          // Frosted, not merely translucent. The composer sits above the thread
          // (see the sticky wrapper in chat-thread.tsx), so scrolling back
          // through a conversation moves text behind it, and a solid card
          // clipping that text at a hard edge is the thing this avoids. The blur
          // is what keeps what passes behind from being *readable*: a second
          // column of words under the one you are typing is worse than either a
          // solid card or a clear one. The saturate is the rest of the effect —
          // blurring alone greys out what it averages, and real frosted glass
          // keeps the colour of what is behind it.
          "bg-background/75 backdrop-blur-2xl backdrop-saturate-150",
          "focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50",
          disabled && "opacity-70"
        )}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* The fuzzy finder is the top half of this card rather than a panel
            floating over it — one outline around both, with a rule where they
            meet. Always mounted so it can play its own exit animation; it
            collapses to nothing once `open` has been false long enough. The
            `overflow-hidden` above is what keeps its top corners on the card's
            radius. */}
        <SlashCommandMenu
          open={isSlashActive}
          input={editorText}
          selectedIndex={selectedIndex}
          onSelectCommand={handleSelectCommand}
          onSelectOption={handleSelectOption}
          ctx={commandContext}
        />

        {/* Drop target feedback. pointer-events-none so it never steals the
            dragleave that would otherwise end the drag it's announcing. */}
        {isDraggingFiles && (
          <div className="pointer-events-none absolute inset-0 z-40 flex items-center justify-center gap-2 rounded-[0.875rem] border-2 border-dashed border-primary bg-primary/5 text-sm font-medium text-primary">
            <Paperclip className="size-4" />
            Drop to add
          </div>
        )}

        <AttachmentStrip
          attachments={attachments}
          onRemove={removeAttachment}
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
          <AttachMenu onFiles={addFiles} disabled={disabled} />
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

      {title !== undefined && (
        <ComposerFooter
          title={title}
          onRenameTitle={onRenameTitle}
          contextTokensUsed={contextTokensUsed}
          contextWindow={contextWindow}
          // The same not-quite-opaque surface as the card above it: the footer
          // is outside the card's outline but on the same floating plane, and a
          // clear strip under a translucent card would put sharp thread text
          // directly beneath blurred thread text.
          className="rounded-lg bg-background/75 py-1 backdrop-blur-md"
        />
      )}
    </div>
  )
}
