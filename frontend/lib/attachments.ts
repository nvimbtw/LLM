import { apiFetch } from "./api"

/**
 * A file stored against a turn. Metadata only — the bytes are a separate,
 * authenticated request, so nothing that merely lists attachments pays for
 * them (see `use-attachment-url.ts`).
 */
export interface Attachment {
  id: string
  filename: string
  mediaType: string
  size: number
}

/** What the API returns per attachment; snake_case, like every other wire type. */
export interface AttachmentWire {
  attachment_id: string
  filename: string
  media_type: string
  size: number
}

export function mapAttachment(raw: AttachmentWire): Attachment {
  return {
    id: raw.attachment_id,
    filename: raw.filename,
    mediaType: raw.media_type,
    size: raw.size,
  }
}

/**
 * How many files one turn may carry. The engine refuses an eleventh
 * (`MAX_PER_MESSAGE` in `engine/api/routes/attachments.py`) — repeated here so
 * it is said in the composer, where the person is still choosing, rather than
 * as a failed turn after the socket has opened.
 */
export const MAX_ATTACHMENTS_PER_MESSAGE = 5

/**
 * Store a file and get back the row it became.
 *
 * Uploaded when it is picked rather than when the turn is sent, because the
 * composer on the home screen has no conversation yet — an attachment belongs
 * to its owner until a turn claims it, which is what makes "attach a file, then
 * start a chat with it" one ordinary path rather than a special case.
 */
export async function uploadAttachment(file: File): Promise<Attachment> {
  const body = new FormData()
  body.append("file", file)

  const response = await apiFetch("/attachments", { method: "POST", body })
  if (!response.ok) {
    throw new Error(await refusal(response, `${file.name} could not be added.`))
  }
  return mapAttachment((await response.json()) as AttachmentWire)
}

/**
 * Drop a staged upload. Only works before a turn has claimed it, which is
 * exactly the window the composer's remove button lives in.
 */
export async function discardAttachment(id: string): Promise<void> {
  const response = await apiFetch(`/attachments/${id}`, { method: "DELETE" })
  if (!response.ok && response.status !== 404) {
    throw new Error("The attachment could not be removed.")
  }
}

/** Where the bytes are. Not usable as an `<img src>` — the request needs a token. */
export function attachmentPath(id: string): string {
  return `/attachments/${id}`
}

export function isImage(attachment: Attachment): boolean {
  return attachment.mediaType.startsWith("image/")
}

/**
 * How an attachment can be shown, which is not the same question as what it is.
 *
 * `embed` is "hand it to the browser and let it render its own viewer" — only
 * PDF, because it is the one format every browser has a viewer for and nothing
 * here is going to grow one.
 */
export type PreviewKind = "image" | "text" | "embed" | "none"

/** Media types that are text without saying `text/`. */
const TEXTUAL = [
  "application/json",
  "application/xml",
  "application/javascript",
  "application/x-yaml",
  "application/x-sh",
]

/**
 * Extensions trusted over the media type, because a browser guesses one from
 * the extension in the first place and guesses badly: `.bak` arrives as
 * `application/x-trash`, which is a verdict rather than a type.
 */
const TEXT_EXTENSIONS = [
  "bak",
  "cfg",
  "conf",
  "csv",
  "diff",
  "env",
  "ini",
  "log",
  "md",
  "patch",
  "rs",
  "py",
  "ts",
  "tsx",
  "js",
  "jsx",
  "toml",
  "tsv",
  "txt",
  "yaml",
  "yml",
]

export function previewKind(attachment: Attachment): PreviewKind {
  const type = attachment.mediaType.toLowerCase()
  if (type.startsWith("image/")) return "image"
  if (type === "application/pdf") return "embed"
  if (type.startsWith("text/") || TEXTUAL.includes(type)) return "text"
  if (TEXT_EXTENSIONS.includes(extensionOf(attachment.filename))) return "text"
  return "none"
}

/** Lowercased and without the dot; empty when the name carries none. */
export function extensionOf(filename: string): string {
  const at = filename.lastIndexOf(".")
  // A leading dot is the whole name of a dotfile, not an extension on nothing.
  if (at <= 0 || at === filename.length - 1) return ""
  return filename.slice(at + 1).toLowerCase()
}

/**
 * The name without its extension.
 *
 * Split so a tile can truncate the long half and keep the short one: a card
 * reading "Frank Herberts Dune Saga Collec…" has thrown away the one part that
 * says what the file *is*, which is the part that fits.
 */
export function stemOf(filename: string): string {
  const extension = extensionOf(filename)
  return extension ? filename.slice(0, -(extension.length + 1)) : filename
}

export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/** The server's own reason, which says what the limit was, or a fallback. */
async function refusal(response: Response, fallback: string): Promise<string> {
  try {
    const body = (await response.json()) as { error?: string }
    return body.error || fallback
  } catch {
    return fallback
  }
}
