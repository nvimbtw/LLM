"use client"

import * as React from "react"
import { ArrowUp, Paperclip, Square } from "lucide-react"
import { EditorContent, useEditor } from "@tiptap/react"
import { Extension, Mark, type Editor, type JSONContent } from "@tiptap/core"
import type { ResolvedPos } from "@tiptap/pm/model"
import StarterKit from "@tiptap/starter-kit"
import Placeholder from "@tiptap/extension-placeholder"
import { useTheme } from "next-themes"
import { useRouter } from "next/navigation"

import { cn } from "@/lib/utils"
import {
  discardAttachment,
  uploadAttachment,
  MAX_ATTACHMENTS_PER_MESSAGE,
  type Attachment,
} from "@/lib/attachments"
import { Button } from "@/components/ui/button"
import { FormError } from "@/components/ui/error-state"
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
import { useSpeechRecognition } from "@/hooks/use-speech-recognition"
import { useDraftTokens } from "@/hooks/use-draft-tokens"
import { ModelSelector } from "./model-selector"
import { BudgetSelector, type ThinkBudget } from "./budget-selector"
import { useComposerChoice } from "./composer-choice-provider"
import { SlashCommandMenu } from "./slash-command-menu"
import { AttachMenu } from "./attach-menu"
import { MicButton } from "./mic-button"
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

// The tail of a dictation the recogniser has not settled on yet. It exists to
// be seen — text firming up as you keep speaking is how you know the microphone
// is keeping up — and to be gone before anything is sent. Both are handled:
// `docToMarkdown` ignores marks it does not know, and the mark comes off when
// dictation ends. `inclusive: false` keeps it from spreading onto characters
// typed against its right edge.
const InterimTranscript = Mark.create({
  name: "interimTranscript",
  inclusive: false,
  renderHTML() {
    return ["span", { class: "opacity-55" }, 0]
  },
})

// Set on the transactions this component dispatches for dictation, so the
// position mapping below can tell them from edits a person made.
const VOICE_TRANSACTION = "voiceTranscript"

export function ChatComposer({
  onSend,
  disabled = false,
  onCancel,
  placeholder = "Message Slate…",
  title,
  onRenameTitle,
  contextTokensUsed = 0,
  unseenTokens = 0,
  draft,
  onPruneContext,
}: {
  onSend: (
    content: string,
    modelId: string,
    budget: ThinkBudget,
    // The uploads this turn claims. Rows already, not files — see
    // `lib/attachments.ts` for why they are stored before the turn exists.
    attachments: Attachment[]
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
  // See composer-footer.tsx — `prunedTokens` from `useChat`.
  unseenTokens?: number
  // Text to load into the composer, replacing whatever is in it — the "edit and
  // ask again" action on a user message (see chat-message.tsx). Identified by
  // object identity rather than by the string, so asking to edit the *same*
  // message twice loads it twice; a plain string prop would be a no-op the
  // second time.
  draft?: { text: string }
  // `/prune` — see lib/context-floor.ts. Optional for the same reason
  // `onRenameTitle` is: the new-chat composer has no chat yet to prune.
  onPruneContext?: () => Promise<{ prunedTurns: number; prunedTokens: number }>
}) {
  // Both live above the router (see composer-choice-provider.tsx) rather than
  // here: sending the first message replaces the new-chat composer with the
  // thread's own, and a choice held in this component's state would be reset by
  // that swap — you would pick a model, send, and land in the conversation on
  // whichever model happens to be first in the list.
  const { model, setModel, budget, setBudget } = useComposerChoice()
  // `/new` — one router for every mount of this component, since starting a
  // fresh chat is the same navigation regardless of which conversation (or
  // none) the composer is currently attached to.
  const router = useRouter()
  const [isEmpty, setIsEmpty] = React.useState(true)
  const [isCommand, setIsCommand] = React.useState(false)
  const [editorText, setEditorText] = React.useState("")
  const [selectedIndex, setSelectedIndex] = React.useState(0)
  const [menuDismissed, setMenuDismissed] = React.useState(false)
  // Uploaded as they are picked; what is staged here is the *link* to a turn,
  // which is made when the turn is sent. See `addFiles` below.
  const [attachments, setAttachments] = React.useState<PendingAttachment[]>([])
  const [isDraggingFiles, setIsDraggingFiles] = React.useState(false)

  const { settings, setSetting, setCommandOpen } = useSettings()
  const { models } = useModels()
  const {
    supported: voiceSupported,
    listening,
    transcript,
    onDevice: voiceOnDevice,
    error: voiceError,
    start: startRecognition,
    stop: stopDictation,
  } = useSpeechRecognition()

  // Where in the document the running dictation is writing, or null when
  // nothing is. Document offsets, so they move whenever the document does —
  // which is what the editor's `onTransaction` below keeps up with.
  const voiceRangeRef = React.useRef<{ from: number; to: number } | null>(null)
  const { theme, setTheme } = useTheme()

  const onSendRef = React.useRef(onSend)
  const submitRef = React.useRef<() => void>(() => {})
  const editorRef = React.useRef<ReturnType<typeof useEditor>>(null)

  React.useEffect(() => {
    onSendRef.current = onSend
  })

  // Object URLs (image thumbnails) are only ever revoked in the three callbacks
  // below — kept as the only places that own their lifetime so none leak past a
  // re-render or an unmount.
  //
  // Emptying the strip comes in two kinds and they are not interchangeable.
  // After a send the rows belong to the turn, so only the local previews are
  // released; abandoning them (the `/clear` command) has to take the rows with
  // it, or every discarded draft leaves stored files nothing will ever claim.
  const releaseAttachments = React.useCallback(() => {
    setAttachments((prev) => {
      for (const a of prev) if (a.previewUrl) URL.revokeObjectURL(a.previewUrl)
      return []
    })
  }, [])

  const discardAttachments = React.useCallback(() => {
    setAttachments((prev) => {
      for (const a of prev) {
        if (a.previewUrl) URL.revokeObjectURL(a.previewUrl)
        // Nothing waits on this and nothing recovers from it: the row is the
        // server's to sweep if the request never lands.
        if (a.attachmentId) void discardAttachment(a.attachmentId).catch(() => {})
      }
      return []
    })
  }, [])

  // The current strip, for the two callbacks below that read it without wanting
  // to be rebuilt whenever it changes: `addFiles` needs to know how full it is,
  // and the unmount sweep needs whatever it holds at the moment it runs.
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

  const settle = React.useCallback(
    (key: string, change: Partial<PendingAttachment>) =>
      setAttachments((prev) =>
        prev.map((a) => (a.key === key ? { ...a, ...change } : a))
      ),
    []
  )

  // Picked is uploaded. The alternative — hold the files and send them with the
  // turn — makes the first message of a new chat a different code path from
  // every other one, because there is no conversation to attach to yet.
  const addFiles = React.useCallback(
    (files: FileList | File[]) => {
      const incoming = Array.from(files)
      if (incoming.length === 0) return

      // Past the cap a file becomes a failed tile rather than being dropped:
      // the strip already has a way to say "this one is not going", and
      // silently ignoring half a drop is the version nobody notices.
      const room = MAX_ATTACHMENTS_PER_MESSAGE - attachmentsRef.current.length
      const staged = incoming.map((file, i): PendingAttachment => {
        const isImage = file.type.startsWith("image/")
        return {
          key: crypto.randomUUID(),
          file,
          kind: isImage ? "image" : "file",
          previewUrl: isImage ? URL.createObjectURL(file) : undefined,
          ...(i < room
            ? { status: "uploading" as const }
            : {
                status: "failed" as const,
                error: `Up to ${MAX_ATTACHMENTS_PER_MESSAGE} files per message.`,
              }),
        }
      })
      setAttachments((prev) => [...prev, ...staged])

      for (const pending of staged.slice(0, Math.max(0, room))) {
        uploadAttachment(pending.file)
          .then((stored) =>
            settle(pending.key, { status: "ready", attachmentId: stored.id })
          )
          .catch((err: unknown) =>
            settle(pending.key, {
              status: "failed",
              error: err instanceof Error ? err.message : undefined,
            })
          )
      }
    },
    [settle]
  )

  const removeAttachment = React.useCallback((key: string) => {
    setAttachments((prev) => {
      const target = prev.find((a) => a.key === key)
      if (target?.previewUrl) URL.revokeObjectURL(target.previewUrl)
      if (target?.attachmentId) {
        void discardAttachment(target.attachmentId).catch(() => {})
      }
      return prev.filter((a) => a.key !== key)
    })
  }, [])

  const clearComposer = React.useCallback(() => {
    stopDictation()
    setIsEmpty(true)
    setIsCommand(false)
    setEditorText("")
    setMenuDismissed(false)
    setSelectedIndex(0)
    discardAttachments()
  }, [discardAttachments, stopDictation])

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
      newChat: () => router.push("/"),
      pruneContext: onPruneContext,
    }),
    [
      settings,
      setSetting,
      setCommandOpen,
      model,
      setModel,
      models,
      budget,
      setBudget,
      theme,
      setTheme,
      clearComposer,
      router,
      onPruneContext,
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
      InterimTranscript,
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
    onTransaction: ({ transaction }) => {
      // Keep the dictation's range over the same text when something else edits
      // the document — typing ahead of it, a paste, a command. Our own inserts
      // carry the meta and set the range themselves, so they are skipped.
      const range = voiceRangeRef.current
      if (!range || !transaction.docChanged) return
      if (transaction.getMeta(VOICE_TRANSACTION)) return
      voiceRangeRef.current = {
        from: transaction.mapping.map(range.from),
        to: transaction.mapping.map(range.to),
      }
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

  const beginDictation = React.useCallback(() => {
    const ed = editorRef.current
    if (!ed || disabled) return
    ed.commands.focus()

    // Speaking straight after a finished word would run the two together. The
    // gap goes in once, here, rather than being guessed at on every result.
    const at = ed.state.selection.to
    const $at = ed.state.doc.resolve(at)
    const previous =
      $at.parentOffset > 0
        ? $at.parent.textBetween($at.parentOffset - 1, $at.parentOffset)
        : ""
    if (previous && !/\s/.test(previous)) {
      ed.commands.insertContentAt(at, { type: "text", text: " " })
    }

    const anchor = ed.state.selection.to
    voiceRangeRef.current = { from: anchor, to: anchor }
    startRecognition()
  }, [disabled, startRecognition])

  // The spoken text, written in as one span that is replaced whole on every
  // event rather than appended to. The interim tail is not additive — the
  // recogniser rewrites its guess as it hears more — and appending successive
  // guesses is exactly how you end up with a paragraph of stuttered half-words.
  React.useEffect(() => {
    const ed = editorRef.current
    const range = voiceRangeRef.current
    if (!ed || !range) return

    const { final, interim } = transcript
    if (!final && !interim) return

    const content: JSONContent[] = []
    if (final) content.push({ type: "text", text: final })
    if (interim) {
      content.push({
        type: "text",
        text: final ? ` ${interim}` : interim,
        marks: [{ type: InterimTranscript.name }],
      })
    }
    const length = content.reduce((n, node) => n + (node.text?.length ?? 0), 0)

    ed.chain()
      .command(({ tr }) => {
        tr.setMeta(VOICE_TRANSACTION, true)
        return true
      })
      .insertContentAt({ from: range.from, to: range.to }, content)
      .setTextSelection(range.from + length)
      .run()

    voiceRangeRef.current = { from: range.from, to: range.from + length }
  }, [transcript])

  // Dictation over: everything in the range is settled now, so the interim
  // styling comes off it. Declared after the effect above so that within one
  // commit the last result to land is written before this runs.
  const wasListeningRef = React.useRef(false)
  React.useEffect(() => {
    const stopped = wasListeningRef.current && !listening
    wasListeningRef.current = listening
    if (!stopped) return

    const ed = editorRef.current
    const range = voiceRangeRef.current
    voiceRangeRef.current = null
    if (!ed || !range || range.to <= range.from) return
    // The text may not be there any more — `/clear` and a send both empty the
    // composer on the way to stopping dictation, and a range reaching past the
    // end of the document is one there is nothing left to unmark.
    if (range.to > ed.state.doc.content.size) return
    ed.chain()
      .setTextSelection(range)
      .unsetMark(InterimTranscript.name)
      .setTextSelection(range.to)
      .focus()
      .run()
  }, [listening])

  // Dictate without reaching for the microphone. Ctrl/Cmd+Shift+Space because
  // the letters that would read better — V, D, M — are all spoken for by the
  // browsers this feature runs in.
  React.useEffect(() => {
    if (!voiceSupported || disabled) return

    const handleShortcut = (event: KeyboardEvent) => {
      if (event.code !== "Space" || !event.shiftKey) return
      if (!event.metaKey && !event.ctrlKey) return
      event.preventDefault()
      if (listening) stopDictation()
      else beginDictation()
    }

    document.addEventListener("keydown", handleShortcut)
    return () => document.removeEventListener("keydown", handleShortcut)
  }, [voiceSupported, disabled, listening, stopDictation, beginDictation])

  const submit = React.useCallback(() => {
    if (!editor || editor.isEmpty || disabled) return

    // Sending ends dictation: what was said is in the message now, and the
    // range it was writing into is about to stop existing. Cleared here rather
    // than left to the effect above, which would otherwise reach for it after
    // the content is gone.
    stopDictation()
    voiceRangeRef.current = null

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

    // Only the ones that made it. A failed upload is left visible rather than
    // sent as nothing — but `canSend` already keeps the button off while any is
    // in flight, so this drops nothing a person is still waiting on.
    const sending: Attachment[] = attachments
      .filter((a) => a.status === "ready" && a.attachmentId)
      .map((a) => ({
        id: a.attachmentId as string,
        filename: a.file.name,
        mediaType: a.file.type,
        size: a.file.size,
      }))

    editor.commands.clearContent()
    setIsEmpty(true)
    setEditorText("")
    releaseAttachments()
    onSendRef.current(markdown, model, budget, sending)
  }, [
    editor,
    disabled,
    model,
    budget,
    attachments,
    releaseAttachments,
    stopDictation,
  ])

  React.useEffect(() => {
    submitRef.current = submit
  })

  React.useEffect(() => {
    editor?.setEditable(!disabled)
  }, [editor, disabled])

  // Load an incoming draft. Built as an explicit document of plain-text
  // paragraphs rather than handed to `setContent` as a string, because a string
  // is parsed as HTML: a question containing `<thing>` would lose it, and the
  // markdown a message is stored as would be interpreted instead of typed.
  // Every line lands as literal text, so `docToMarkdown` on submit gives back
  // exactly what was loaded.
  //
  // No state is set here — `setContent` emits the editor's own update, and
  // `onUpdate` above already owns `isEmpty` / `editorText` / `isCommand`.
  React.useEffect(() => {
    const ed = editorRef.current
    if (!ed || !draft) return
    ed.commands.setContent({
      type: "doc",
      content: draft.text.split("\n").map((line) => ({
        type: "paragraph",
        ...(line ? { content: [{ type: "text", text: line }] } : {}),
      })),
    })
    ed.commands.focus("end")
  }, [draft])

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

  // Nothing goes while an upload is in flight: the turn would be stored without
  // the file the person is watching arrive, and there is no second chance to
  // attach it to a message already written.
  const uploading = attachments.some((a) => a.status === "uploading")
  const canSend = !isEmpty && !disabled && !uploading

  const contextWindow =
    models.find((m) => m.id === model)?.contextWindow ?? null

  // The markdown, not the text on screen: the `**` and the `- ` are in the
  // string the model is handed, and on a 256-token window a list's bullets are
  // a percentage rather than a rounding error. A recognised standalone command
  // never reaches the model at all, so it costs the window nothing — `submit`
  // runs it locally and clears the composer.
  const readDraft = () => {
    const ed = editorRef.current
    if (!ed || matchStandaloneCommand(ed.getText())) return ""
    return docToMarkdown(ed.getJSON())
  }
  const draftTokens = useDraftTokens(editorText, model, readDraft)

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
    // The composer card, and the footer (see composer-footer.tsx) as a drawer
    // pulled out from under it: inset either side, rounded along the bottom
    // only, and tucked up behind the card so there is no seam between them.
    //
    // They used to be two elements 6px apart, and since the composer is frosted
    // and the thread scrolls behind it, that 6px was a slot of sharp text
    // running between two blurred surfaces. Attaching the drawer closes the slot
    // without making the footer a compartment of the card — which it is not: it
    // says what the conversation is, not what you are about to send.
    <div className="flex flex-col">
      <div
        className={cn(
          // Above the drawer, and opaque enough to hide the part of it that
          // reaches up behind the card.
          "z-10",
          // A hairline, not a form-control boundary: `border-input` is tuned to
          // 3:1 because it has to say "click here" on a bare page, and around
          // the largest element in the app that weight reads as a frame drawn
          // around the whole screen. The composer says what it is by being a
          // raised sheet (the card surface below) — the edge only has to close
          // the shape.
          "relative overflow-hidden rounded-(--radius-composer) border shadow-raised transition-colors",
          // Frosted, not merely translucent. The composer sits above the thread
          // (see the sticky wrapper in chat-thread.tsx), so scrolling back
          // through a conversation moves text behind it, and a solid card
          // clipping that text at a hard edge is the thing this avoids. The blur
          // is what keeps what passes behind from being *readable*: a second
          // column of words under the one you are typing is worse than either a
          // solid card or a clear one. The saturate is the rest of the effect —
          // blurring alone greys out what it averages, and real frosted glass
          // keeps the colour of what is behind it.
          // The card surface rather than the page's: the composer is a sheet
          // resting on the thread, and taking the page's own colour made it a
          // hole in the page that only its outline distinguished.
          "bg-card/75 backdrop-blur-2xl backdrop-saturate-150",
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
          <div className="pointer-events-none absolute inset-0 z-40 flex items-center justify-center gap-2 rounded-(--radius-composer) border-2 border-dashed border-primary bg-primary/5 text-sm font-medium text-primary">
            <Paperclip className="size-4" />
            Drop to add
          </div>
        )}

        <AttachmentStrip attachments={attachments} onRemove={removeAttachment} />

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
        {voiceError && (
          <FormError className="px-3.5 pb-1.5">{voiceError}</FormError>
        )}

        <div className="flex items-center gap-2 px-2.5 pb-2">
          <AttachMenu onFiles={addFiles} disabled={disabled} />
          <ModelSelector value={model} onValueChange={setModel} />
          <BudgetSelector value={budget} onValueChange={setBudget} />
          {/* The two ways a message leaves this box, kept together at the
              right end of the row. The microphone is simply absent where the
              browser has no recogniser — a control that cannot work is worse
              than no control. */}
          <div className="ml-auto flex items-center gap-2">
            {voiceSupported && (
              <MicButton
                listening={listening}
                onDevice={voiceOnDevice}
                onStart={beginDictation}
                onStop={stopDictation}
                disabled={disabled}
              />
            )}
            <Button
              aria-label={disabled && onCancel ? "Stop" : "Send"}
              size="icon-sm"
              className="rounded-full"
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
      </div>

      {title !== undefined && (
        <ComposerFooter
          title={title}
          onRenameTitle={onRenameTitle}
          contextTokensUsed={contextTokensUsed}
          draftTokens={draftTokens}
          contextWindow={contextWindow}
          contextMultiplier={settings.contextMultiplier}
          unseenTokens={unseenTokens}
          listening={listening}
          onDeviceDictation={voiceOnDevice}
          // The drawer. `mx-2` is the inset that makes it read as a thing under
          // the composer rather than a continuation of it; `-mt-3` reaches it
          // back up behind the card, so the two rounded shapes meet with no
          // notch of thread showing between them. No border, and a surface
          // faint enough to be invisible against the page — it only resolves
          // into a plate when text passes under it, which is the whole job it
          // has to do.
          //
          // The padding is asymmetric *so that it looks symmetric*: 12px of the
          // top is hidden behind the card, so `pt-4` (16px) leaves 4px of
          // visible space above the text and `pb-1` gives it the same below.
          // Centring the box would put the text high by exactly the overlap.
          // Keep the two in step — `pt` is always the overlap plus `pb`.
          className="mx-2 -mt-3 rounded-b-(--radius-composer) bg-card/60 px-2 pt-4 pb-1 backdrop-blur-md"
        />
      )}
    </div>
  )
}
