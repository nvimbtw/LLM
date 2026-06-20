"use client"

import * as React from "react"

import { API_URL, getToken } from "@/lib/api"
import { hasAcceptedExtension } from "@/components/files/constants"

export type UploadStatus = "idle" | "uploading" | "success" | "error"

export interface QueuedFile {
  id: string
  file: File
}

let counter = 0
function nextId() {
  counter += 1
  return `${Date.now()}-${counter}`
}

export function useFileUpload() {
  const [files, setFiles] = React.useState<QueuedFile[]>([])
  const [status, setStatus] = React.useState<UploadStatus>("idle")
  const [progress, setProgress] = React.useState(0)
  const [error, setError] = React.useState<string | null>(null)

  // Returns the names that were rejected for an unsupported format, so the
  // caller can surface them. Accepted files are de-duplicated by name + size.
  function addFiles(incoming: FileList | File[]): string[] {
    const list = Array.from(incoming)
    const rejected: string[] = []
    const accepted: File[] = []

    for (const file of list) {
      if (!hasAcceptedExtension(file.name)) {
        rejected.push(file.name)
        continue
      }
      accepted.push(file)
    }

    if (accepted.length > 0) {
      setStatus("idle")
      setError(null)
      setFiles((prev) => {
        const seen = new Set(prev.map((q) => `${q.file.name}:${q.file.size}`))
        const next = accepted
          .filter((f) => !seen.has(`${f.name}:${f.size}`))
          .map((file) => ({ id: nextId(), file }))
        return [...prev, ...next]
      })
    }

    return rejected
  }

  function removeFile(id: string) {
    setFiles((prev) => prev.filter((q) => q.id !== id))
  }

  function clear() {
    setFiles([])
    setStatus("idle")
    setProgress(0)
    setError(null)
  }

  // Sends every queued file as multipart/form-data under the `files` field.
  // Uses XHR rather than fetch so we can report real upload progress. We set
  // only the Authorization header — the browser fills in the multipart
  // Content-Type (with boundary) itself.
  function upload(): Promise<void> {
    if (files.length === 0) {
      return Promise.reject(new Error("No files selected."))
    }

    const form = new FormData()
    for (const { file } of files) {
      form.append("files", file, file.name)
    }

    setStatus("uploading")
    setProgress(0)
    setError(null)

    return new Promise<void>((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open("POST", `${API_URL}/files`)

      const token = getToken()
      if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`)

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          setProgress(Math.round((event.loaded / event.total) * 100))
        }
      }

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          setStatus("success")
          setProgress(100)
          setFiles([])
          resolve()
        } else {
          const message = xhr.responseText || "Upload failed."
          setStatus("error")
          setError(message)
          reject(new Error(message))
        }
      }

      xhr.onerror = () => {
        const message = "Network error during upload."
        setStatus("error")
        setError(message)
        reject(new Error(message))
      }

      xhr.send(form)
    })
  }

  return { files, status, progress, error, addFiles, removeFile, clear, upload }
}
