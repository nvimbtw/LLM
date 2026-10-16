"use client"

import Editor, { loader } from "@monaco-editor/react"
import { useTheme } from "next-themes"

loader.config({ paths: { vs: "/monaco/vs" } })

export interface PythonSourceEditorProps {
  value: string
  readOnly: boolean
  onChange: (value: string) => void
  height?: string
}

/** Displays locally hosted Python syntax highlighting and optional full-file editing. */
export function PythonSourceEditor({
  value,
  readOnly,
  onChange,
  height = "360px",
}: PythonSourceEditorProps) {
  const { resolvedTheme } = useTheme()
  return (
    <Editor
      height={height}
      language="python"
      value={value}
      theme={resolvedTheme === "dark" ? "vs-dark" : "vs"}
      onChange={(next) => onChange(next ?? "")}
      options={{
        readOnly,
        minimap: { enabled: false },
        fontSize: 12,
        scrollBeyondLastLine: false,
        wordWrap: "on",
        tabSize: 4,
        ariaLabel: "Processor Python source",
        automaticLayout: true,
      }}
    />
  )
}
