"use client"

import * as React from "react"
import { ArrowUp, Paperclip, Square } from "lucide-react"
import { EditorContent } from "@tiptap/react"
import type { Editor } from "@tiptap/core"
import { useTheme } from "next-themes"
import { useRouter } from "next/navigation"

import { cn } from "@/lib/utils"
import type { Attachment } from "@/lib/attachments"
import { readyAttachments } from "@/lib/composer-attachments"
import { useComposerAttachments } from "@/hooks/use-composer-attachments"
import { useComposerHeight } from "@/hooks/use-composer-height"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { FormError } from "@/components/ui/error-state"
import { docToMarkdown } from "@/lib/tiptap-markdown"
import {
  matchStandaloneCommand,
  runSlashCommand,
  type CommandContext,
} from "@/lib/commands"
import { useSettings } from "@/components/settings/settings-provider"
import { useModels } from "@/hooks/use-models"
import { useComposerDictation } from "@/hooks/use-composer-dictation"
import {
  useComposerEditor,
  type ComposerEditorSnapshot,
} from "@/hooks/use-composer-editor"
import { useComposerCommands } from "@/hooks/use-composer-commands"
import { useDraftTokens } from "@/hooks/use-draft-tokens"
import { SectionTexture } from "@/components/ui/section-texture"
import { ComposerPicker } from "./composer-picker"
import type { ThinkBudget } from "@/lib/think-budget"
import { useComposerChoice } from "./composer-choice-provider"
import { SlashCommandMenu } from "./slash-command-menu"
import { AttachMenu } from "./attach-menu"
import { MicButton } from "./mic-button"
import { AttachmentStrip } from "./attachment-strip"
import { ComposerFooter } from "./composer-footer"
import { ComposerPlan } from "./composer-plan"
import type { Plan } from "@/lib/plan"
import { usePlanEditor } from "@/hooks/use-plan-editor"
import { useLocalPlan } from "@/hooks/use-local-plan"
import { effectiveContextMultiplier } from "@/lib/model-context"

export interface ChatComposerProps {
  plan?: Plan | null
  onPlanChange?: (plan: Plan) => void
  onPlanEditingChange?: (editing: boolean) => void
  onSend: (
    content: string,
    modelId: string,
    budget: ThinkBudget,
    attachments: Attachment[]
  ) => Promise<void> | void
  disabled?: boolean
  sending?: boolean
  /** Uses a smaller resting editor once a conversation has messages. */
  compact?: boolean
  onCancel?: () => void
  placeholder?: string
  title?: string
  onRenameTitle?: (title: string) => void | Promise<void>
  contextTokensUsed?: number
  unseenTokens?: number
  draft?: { text: string }
  onPruneContext?: () => Promise<{ prunedTurns: number; prunedTokens: number }>
}

/** Composes the editor, commands, upload draft, dictation and footer; onSend receives markdown and completed uploads. */
export function ChatComposer({
  onSend,
  disabled = false,
  sending = false,
  compact = false,
  onCancel,
  placeholder = "Message Slate…",
  contextTokensUsed = 0,
  unseenTokens = 0,
  draft,
  onPruneContext,
  plan,
  onPlanChange,
  onPlanEditingChange,
}: ChatComposerProps) {
  const fallbackPlan = useLocalPlan()
  const savedPlan = plan === undefined ? fallbackPlan.plan : plan
  const planEditor = usePlanEditor(
    savedPlan,
    onPlanChange ?? fallbackPlan.save,
    onPlanEditingChange
  )
  const visiblePlan = planEditor.editing?.draft ?? savedPlan
  const {
    model,
    setModel,
    budget,
    setBudget,
    isLoading: choiceLoading,
    modelFacts,
  } = useComposerChoice()
  const router = useRouter()
  const [isEmpty, setIsEmpty] = React.useState(true)
  const [isCommand, setIsCommand] = React.useState(false)
  const [editorText, setEditorText] = React.useState("")
  const { shellRef, bodyRef } = useComposerHeight()
  const {
    attachments,
    addFiles,
    removeAttachment,
    releaseAttachments,
    discardAttachments,
  } = useComposerAttachments()
  const [isDraggingFiles, setIsDraggingFiles] = React.useState(false)

  const { settings, setSetting, setCommandOpen } = useSettings()
  const { models, isLoading: modelsLoading } = useModels()
  const { theme, setTheme } = useTheme()

  const onSendRef = React.useRef(onSend)
  const submitRef = React.useRef<() => void>(() => {})
  const editorRef = React.useRef<Editor | null>(null)
  const {
    supported: voiceSupported,
    listening,
    onDevice: voiceOnDevice,
    error: voiceError,
    begin: beginDictation,
    stop: stopDictation,
    releaseRange,
    mapTransaction,
  } = useComposerDictation(editorRef, disabled)

  React.useEffect(() => {
    onSendRef.current = onSend
  })

  /** Abandons the current input and uploads, stopping dictation before the command clears editor content. */
  const clearComposer = React.useCallback(() => {
    stopDictation()
    setIsEmpty(true)
    setIsCommand(false)
    setEditorText("")
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
      plan: planEditor,
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
      planEditor,
    ]
  )

  const commandContextRef = React.useRef(commandContext)
  React.useEffect(() => {
    commandContextRef.current = commandContext
  })

  const commandMenu = useComposerCommands(
    editorRef,
    commandContext,
    editorText,
    setEditorText
  )
  const {
    selectedIndex,
    isSlashActive,
    handleSelectCommand,
    handleSelectOption,
  } = commandMenu
  const commandMenuRef = React.useRef(commandMenu)
  React.useEffect(() => {
    commandMenuRef.current = commandMenu
  })

  /** Reflects editor updates in presentation state without owning editor transactions. */
  const updateEditorState = React.useCallback(
    (snapshot: ComposerEditorSnapshot) => {
      setIsEmpty(snapshot.isEmpty)
      setIsCommand(snapshot.isCommand)
      setEditorText(snapshot.text)
    },
    []
  )
  const editor = useComposerEditor({
    editorRef,
    submitRef,
    commandMenuRef,
    onChange: updateEditorState,
    mapTransaction,
    placeholder,
    disabled,
    draft,
  })

  /** Runs standalone commands locally; ordinary turns send markdown only after uploads and model choice are ready. */
  const submit = React.useCallback(() => {
    if (!editor || editor.isEmpty || disabled) return

    stopDictation()
    releaseRange()

    const text = editor.getText()
    if (matchStandaloneCommand(text)) {
      runSlashCommand(text, commandContextRef.current)
      editor.commands.clearContent()
      commandMenuRef.current.setMenuDismissed(false)
      setIsEmpty(true)
      setIsCommand(false)
      setEditorText("")
      return
    }

    if (attachments.some((item) => item.status === "uploading")) return
    if (choiceLoading || modelsLoading || !model) return
    const markdown = docToMarkdown(editor.getJSON())
    if (!markdown.trim()) return

    const toSend = readyAttachments(attachments)

    editor.commands.clearContent()
    setIsEmpty(true)
    setEditorText("")
    releaseAttachments()
    onSendRef.current(markdown, model, budget, toSend)
  }, [
    editor,
    disabled,
    choiceLoading,
    modelsLoading,
    model,
    budget,
    attachments,
    releaseAttachments,
    stopDictation,
    releaseRange,
  ])

  React.useEffect(() => {
    submitRef.current = submit
  })

  const uploading = attachments.some((a) => a.status === "uploading")
  const canSend =
    !isEmpty &&
    !disabled &&
    !uploading &&
    (isCommand || (!choiceLoading && !modelsLoading && Boolean(model)))

  const selectedModel =
    modelFacts?.id === model
      ? modelFacts
      : models.find((item) => item.id === model)
  const contextWindow = selectedModel?.contextWindow ?? null

  /** Returns the exact outgoing markdown for token counting, or an empty string for local commands. */
  const readDraft = () => {
    const ed = editorRef.current
    if (!ed || matchStandaloneCommand(ed.getText())) return ""
    return docToMarkdown(ed.getJSON())
  }
  const draftTokens = useDraftTokens(editorText, model, readDraft)

  /** Identifies external file drags while the composer is enabled. */
  const isFileDrag = (e: React.DragEvent) =>
    !disabled && Boolean(e.dataTransfer?.types.includes("Files"))

  const dragDepthRef = React.useRef(0)

  /** Shows the file-drop overlay when a file drag first enters this composer. */
  const handleDragEnter = (e: React.DragEvent) => {
    if (!isFileDrag(e)) return
    e.preventDefault()
    dragDepthRef.current += 1
    setIsDraggingFiles(true)
  }
  /** Keeps enabled file drops accepted while the pointer crosses child elements. */
  const handleDragOver = (e: React.DragEvent) => {
    if (!isFileDrag(e)) return
    e.preventDefault()
  }
  /** Hides the overlay only after the file drag leaves every composer child. */
  const handleDragLeave = (e: React.DragEvent) => {
    if (!isFileDrag(e)) return
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1)
    if (dragDepthRef.current === 0) setIsDraggingFiles(false)
  }
  /** Clears drag feedback and stages the dropped files for immediate upload. */
  const handleDrop = (e: React.DragEvent) => {
    if (!isFileDrag(e)) return
    e.preventDefault()
    dragDepthRef.current = 0
    setIsDraggingFiles(false)
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files)
  }

  return (
    <div className="flex flex-col">
      <div className="relative z-20">
        <div
          ref={shellRef}
          data-composer-size={compact ? "compact" : "expanded"}
          className={cn(
            "z-10",
            "relative overflow-hidden rounded-(--radius-composer) border shadow-raised transition-[height] duration-200 ease-out motion-reduce:transition-none",
            "bg-card/75 backdrop-blur-2xl backdrop-saturate-150",
            disabled && "opacity-70"
          )}
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div ref={bodyRef}>
            {(isSlashActive || visiblePlan) && (
              <div className="relative isolate overflow-hidden">
                <SectionTexture />
                {visiblePlan && (
                  <div hidden={isSlashActive}>
                    <ComposerPlan
                      key={visiblePlan.id}
                      plan={visiblePlan}
                      running={sending}
                      editing={planEditor.editing}
                    />
                  </div>
                )}
                <SlashCommandMenu
                  open={isSlashActive}
                  input={editorText}
                  selectedIndex={selectedIndex}
                  onSelectCommand={handleSelectCommand}
                  onSelectOption={handleSelectOption}
                  ctx={commandContext}
                />
              </div>
            )}
            {isDraggingFiles && (
              <div className="pointer-events-none absolute inset-0 z-40 flex items-center justify-center gap-2 rounded-(--radius-composer) border-2 border-dashed border-primary bg-primary/5 text-sm font-medium text-primary">
                <Paperclip className="size-4" />
                Drop to add
              </div>
            )}

            <AttachmentStrip
              attachments={attachments}
              onRemove={removeAttachment}
            />

            <div
              className={cn(
                compact
                  ? "[&_.tiptap]:min-h-0 [&_.tiptap]:py-2"
                  : "[&_.tiptap]:min-h-16",
                isCommand &&
                  "[&_.tiptap]:font-medium [&_.tiptap]:text-primary [&_.tiptap]:caret-primary"
              )}
            >
              {editor ? (
                <EditorContent editor={editor} />
              ) : (
                <div
                  className={cn("px-3.5", compact ? "py-2" : "min-h-16 py-3")}
                  role="status"
                >
                  <span className="sr-only">Loading message editor</span>
                  <Skeleton
                    aria-hidden="true"
                    className="h-5 w-40 motion-reduce:animate-none"
                  />
                </div>
              )}
            </div>
            {voiceError && (
              <FormError className="px-3.5 pb-1.5">{voiceError}</FormError>
            )}

            <div className="flex items-center gap-2 px-2.5 pb-2">
              <AttachMenu onFiles={addFiles} disabled={disabled || sending} />
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
                {sending && onCancel && !disabled && (
                  <Button
                    aria-label="Stop"
                    size="icon-sm"
                    variant="ghost"
                    className="rounded-full"
                    onClick={onCancel}
                  >
                    <Square className="size-3.5 fill-current" />
                  </Button>
                )}
                <Button
                  aria-label="Send"
                  size="icon-sm"
                  className="rounded-full"
                  disabled={!canSend}
                  onClick={submit}
                >
                  <ArrowUp />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <ComposerFooter
        choices={
          <ComposerPicker
            model={model}
            onModelChange={setModel}
            budget={budget}
            onBudgetChange={setBudget}
          />
        }
        contextTokensUsed={contextTokensUsed}
        draftTokens={draftTokens}
        contextWindow={contextWindow}
        isLoading={modelsLoading || choiceLoading}
        contextMultiplier={effectiveContextMultiplier(
          selectedModel,
          settings.contextMultiplier
        )}
        unseenTokens={unseenTokens}
        listening={listening}
        onDeviceDictation={voiceOnDevice}
        className="mx-2 -mt-3 rounded-b-(--radius-composer) bg-card/60 px-2 pt-4 pb-1 backdrop-blur-md"
      />
      {selectedModel &&
        !choiceLoading &&
        settings.contextMultiplier > 1 &&
        !selectedModel.supportsContextExtension && (
          <p className="px-4 pt-1 text-xs text-muted-foreground" role="status">
            This model uses its trained context window; the extension preference
            does not apply.
          </p>
        )}
    </div>
  )
}
