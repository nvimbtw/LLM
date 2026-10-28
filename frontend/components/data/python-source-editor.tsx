"use client"
import { SourceEditor } from "@/components/authoring/source-editor"
export interface PythonSourceEditorProps {
  value: string
  readOnly: boolean
  onChange: (value: string) => void
  height?: string
}
/** Keeps processor source editing behavior while using the neutral locally hosted editor surface. */
export function PythonSourceEditor(props: PythonSourceEditorProps) {
  return (
    <SourceEditor
      {...props}
      language="python"
      ariaLabel="Processor Python source"
    />
  )
}
