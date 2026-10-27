"use client"

import Editor, { loader } from "@monaco-editor/react"
import { useTheme } from "next-themes"

loader.config({ paths: { vs: "/monaco/vs" } })

export interface SourceEditorProps {
  language: "python" | "yaml"
  ariaLabel: string
  value: string
  readOnly: boolean
  onChange: (value: string) => void
  height?: string
  line?: number
}

/** Displays locally hosted Python or YAML source with an explicit accessible name and editing policy. */
export function SourceEditor({
  language,
  ariaLabel,
  value,
  readOnly,
  onChange,
  height = "360px",
  line,
}: SourceEditorProps) {
  const { resolvedTheme } = useTheme()
  return (
    <Editor
      height={height}
      language={language}
      value={value}
      theme={resolvedTheme === "dark" ? "vs-dark" : "vs"}
      onChange={(next) => onChange(next ?? "")}
      onMount={(editor) => {
        if (line == null) return
        const target = Math.min(
          Math.max(1, line),
          editor.getModel()?.getLineCount() ?? 1
        )
        editor.setPosition({ lineNumber: target, column: 1 })
        editor.revealLineInCenter(target)
        editor.focus()
      }}
      options={{
        readOnly,
        minimap: { enabled: false },
        fontSize: 12,
        scrollBeyondLastLine: false,
        wordWrap: "on",
        tabSize: 4,
        ariaLabel,
        automaticLayout: true,
      }}
    />
  )
}
