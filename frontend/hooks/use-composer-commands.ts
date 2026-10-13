"use client"

import * as React from "react"
import type { Editor } from "@tiptap/core"
import {
  getCommandSuggestions,
  runSlashCommand,
  type CommandContext,
  type SlashCommandDef,
} from "@/lib/commands"

type Suggestions = NonNullable<ReturnType<typeof getCommandSuggestions>>

/** Counts selectable command or argument rows, including the single action row. */
function suggestionCount(result: Suggestions | null): number {
  if (!result) return 0
  if (result.mode === "command") return result.suggestions.length
  const { data } = result
  if ("options" in data) return data.options.length
  if (data.type === "number") return data.info.presets.length
  if (data.type === "text") return data.info.templates.length
  return 1
}

/** Resolves a highlighted argument into its command value and whether selection should execute it immediately. */
function selectedArgument(
  result: Suggestions,
  index: number
): { value: string; execute: boolean } | null {
  if (result.mode !== "argument") return null
  const { data } = result
  if (data.type === "model") {
    const option = data.options[index]
    return option ? { value: option.model.id, execute: true } : null
  }
  if ("options" in data) {
    const option = data.options[index]
    return option
      ? {
          value: option.value,
          execute: data.type === "plan" ? data.options[index].execute : true,
        }
      : null
  }
  if (data.type === "number") {
    const preset = data.info.presets[index]
    return preset
      ? { value: preset.value.toFixed(data.info.precision), execute: true }
      : null
  }
  if (data.type === "text") {
    const template = data.info.templates[index]
    return template ? { value: template.value, execute: true } : null
  }
  return null
}

/** Owns slash-menu discovery, selection and local command execution for the current editor text/context. */
export function useComposerCommands(
  editorRef: React.RefObject<Editor | null>,
  context: CommandContext,
  editorText: string,
  setText: (text: string) => void
) {
  const [selectedIndex, setSelectedIndex] = React.useState(0)
  const [menuDismissed, setMenuDismissed] = React.useState(false)
  const contextRef = React.useRef(context)
  React.useEffect(() => {
    contextRef.current = context
  })
  /** Clears command input after local execution, preserving the attachment draft. */
  const resetInput = React.useCallback(() => {
    setText("")
    setMenuDismissed(false)
  }, [setText])
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
    return getCommandSuggestions(editorText, context)
  }, [isSlashActive, editorText, context])

  const suggestionsCount = suggestionCount(suggestionsResult)

  /** Executes an action command locally or completes a command requiring an argument. */
  const handleSelectCommand = React.useCallback(
    (cmd: SlashCommandDef) => {
      const ed = editorRef.current
      if (!ed) return

      if (cmd.kind === "action") {
        runSlashCommand(`/${cmd.name}`, contextRef.current)
        ed.commands.clearContent()
        resetInput()
        return
      }

      ed.commands.setContent(`/${cmd.name} `)
      ed.commands.focus("end")
      setText(`/${cmd.name} `)
      setSelectedIndex(0)
    },
    [editorRef, resetInput, setText]
  )

  /** Executes a selected argument or leaves its completed command in the editor for further text. */
  const handleSelectOption = React.useCallback(
    (optionValue: string, executeImmediately = true) => {
      const ed = editorRef.current
      if (!ed) return

      const text = ed.getText().trim()
      const commandWord = text.slice(1).split(/\s+/)[0]
      const fullCmd = `/${commandWord} ${optionValue}`.trim()

      if (executeImmediately) {
        runSlashCommand(fullCmd, contextRef.current)
        ed.commands.clearContent()
        resetInput()
        return
      }

      ed.commands.setContent(`${fullCmd} `)
      ed.commands.focus("end")
      setText(`${fullCmd} `)
    },
    [editorRef, resetInput, setText]
  )

  /** Executes or completes the highlighted row, returning whether a valid suggestion was selected. */
  const selectCurrentSuggestion = React.useCallback(() => {
    if (!suggestionsResult || suggestionsCount === 0 || !editorRef.current)
      return false
    if (suggestionsResult.mode === "command") {
      const command = suggestionsResult.suggestions[selectedIndex]?.command
      if (!command) return false
      handleSelectCommand(command)
      return true
    }
    if (suggestionsResult.data.type === "action") {
      handleSelectCommand(suggestionsResult.command)
      return true
    }
    const option = selectedArgument(suggestionsResult, selectedIndex)
    if (!option) return false
    handleSelectOption(option.value, option.execute)
    return true
  }, [
    suggestionsResult,
    suggestionsCount,
    editorRef,
    selectedIndex,
    handleSelectCommand,
    handleSelectOption,
  ])

  return {
    selectedIndex,
    setSelectedIndex,
    setMenuDismissed,
    isSlashActive,
    suggestionsCount,
    selectCurrentSuggestion,
    handleSelectCommand,
    handleSelectOption,
  }
}
