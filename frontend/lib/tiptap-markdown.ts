// Serializes a TipTap/ProseMirror document into the markdown subset that
// `parseUserMarkdown` understands, so the composer round-trips cleanly.

interface JSONNode {
  type: string
  attrs?: Record<string, unknown>
  content?: JSONNode[]
  text?: string
  marks?: { type: string }[]
}

function serializeInline(nodes: JSONNode[] = []): string {
  return nodes
    .map((n) => {
      if (n.type === "hardBreak") return "\n"
      if (n.type !== "text") return ""
      const marks = new Set((n.marks ?? []).map((m) => m.type))
      let t = n.text ?? ""
      if (marks.has("code")) return "`" + t + "`" // code wins; no nested markers
      if (marks.has("bold")) t = "**" + t + "**"
      if (marks.has("italic")) t = "*" + t + "*"
      return t
    })
    .join("")
}

function serializeList(list: JSONNode, depth: number): string {
  const ordered = list.type === "orderedList"
  const indent = "  ".repeat(depth)
  const lines: string[] = []
  let idx = (list.attrs?.start as number) ?? 1

  for (const item of list.content ?? []) {
    const blocks = item.content ?? []
    const nested: JSONNode[] = []
    let text = ""
    for (const b of blocks) {
      if (b.type === "bulletList" || b.type === "orderedList") nested.push(b)
      else if (b.type === "paragraph" && text === "")
        text = serializeInline(b.content)
    }
    lines.push(indent + (ordered ? `${idx}. ` : "- ") + text)
    for (const nl of nested) lines.push(serializeList(nl, depth + 1))
    idx++
  }

  return lines.join("\n")
}

function serializeBlocks(nodes: JSONNode[] = []): string[] {
  const out: string[] = []
  for (const n of nodes) {
    switch (n.type) {
      case "paragraph":
        out.push(serializeInline(n.content))
        break
      case "codeBlock": {
        const lang = (n.attrs?.language as string) ?? ""
        const code = (n.content ?? []).map((c) => c.text ?? "").join("")
        out.push("```" + lang + "\n" + code + "\n```")
        break
      }
      case "bulletList":
      case "orderedList":
        out.push(serializeList(n, 0))
        break
      default:
        if (n.content) out.push(...serializeBlocks(n.content))
    }
  }
  return out
}

export function docToMarkdown(doc: { content?: JSONNode[] }): string {
  return serializeBlocks(doc.content)
    .map((b) => b.replace(/[ \t]+$/g, ""))
    .filter((b) => b.trim().length > 0)
    .join("\n\n")
    .trim()
}
