// Supported upload formats. Validation is by file extension (more reliable than
// the browser-reported MIME type, which is inconsistent for .docx in
// particular). The MIME types are still listed for the file picker's `accept`.
export const ACCEPTED_EXTENSIONS = [".txt", ".pdf", ".docx"] as const

export const ACCEPT_ATTR = [
  ".txt",
  ".pdf",
  ".docx",
  "text/plain",
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
].join(",")

export function hasAcceptedExtension(name: string): boolean {
  const lower = name.toLowerCase()
  return ACCEPTED_EXTENSIONS.some((ext) => lower.endsWith(ext))
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B"
  const units = ["B", "KB", "MB", "GB"]
  const i = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1
  )
  const value = bytes / Math.pow(1024, i)
  return `${value.toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}
