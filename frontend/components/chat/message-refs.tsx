"use client"

import * as React from "react"

import { stripThinking } from "@/lib/message-parser"

/**
 * Lets a `<|reply|>` token inside one message find the message it points at.
 *
 * The renderer is handed a single message's tree and knows nothing about the
 * conversation around it, so a reference has to be resolved from outside. A
 * context rather than a prop drilled through every node: `<|reply|>` can appear
 * at any depth, and threading a lookup down through paragraphs and list items to
 * reach it would touch every branch of the renderer for one leaf.
 *
 * Unresolvable ids are expected, not exceptional — a model can emit an id that
 * was never real, or one from a message since deleted, and history loaded from a
 * different page may not hold the target at all. `useMessageRef` returns null and
 * the renderer degrades to a plain marker.
 */
export interface MessageRef {
  role: "user" | "assistant"
  /** One line of the referenced message, already stripped of markup. */
  excerpt: string
}

const MessageRefsContext = React.createContext<
  ((id: string) => MessageRef | null) | null
>(null)

export function MessageRefsProvider({
  lookup,
  children,
}: {
  lookup: (id: string) => MessageRef | null
  children: React.ReactNode
}) {
  return (
    <MessageRefsContext.Provider value={lookup}>
      {children}
    </MessageRefsContext.Provider>
  )
}

export function useMessageRef(id: string): MessageRef | null {
  const lookup = React.useContext(MessageRefsContext)
  // No provider at all is a valid state — a message rendered outside a thread
  // (a preview, a search result) simply cannot resolve references.
  return lookup ? lookup(id) : null
}

// How much of the referenced message to quote. Long enough to recognise which
// message is meant, short enough that the reference never competes with the
// reply that carries it.
const EXCERPT_CHARS = 120

/**
 * Builds the lookup from a list of messages. Reasoning is stripped before the
 * markup is, so a reply that opens with a thinking block excerpts its answer
 * rather than the first line of its private reasoning.
 */
export function buildMessageRefs(
  messages: { id?: string; role: "user" | "assistant"; content: string }[]
): (id: string) => MessageRef | null {
  const byId = new Map<string, MessageRef>()
  for (const message of messages) {
    if (!message.id) continue
    byId.set(message.id, {
      role: message.role,
      excerpt: excerpt(message.content),
    })
  }
  return (id) => byId.get(id) ?? null
}

/**
 * The DOM id a message is anchored under, so a `<|reply|>` can scroll to it.
 *
 * Prefixed rather than using the raw message id: these ids come from the server
 * and land in `document.getElementById`, where an unprefixed value could collide
 * with any other element on the page.
 */
export function messageDomId(id: string): string {
  return `msg-${id}`
}

/**
 * Scrolls the referenced message into view, then sweeps its border once you
 * arrive.
 *
 * The sweep is the important half. Scrolling alone lands you somewhere in the
 * thread with no indication of *which* message you were sent to — especially when
 * the target was already on screen and nothing appears to happen at all.
 *
 * It fires on arrival rather than on click: an animation that plays while the
 * page is still travelling is over by the time you get there, which is the one
 * moment it needed to be visible.
 *
 * Written directly to the DOM because the target is another message's subtree,
 * possibly far up the list; lifting "which message is highlighted" into React
 * state would re-render the whole thread to animate one border.
 */
export function revealMessage(id: string): boolean {
  const el = document.getElementById(messageDomId(id))
  if (!el) return false

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches

  const sweep = () => {
    // Restart the animation if it is already running — clicking the same
    // reference twice should sweep twice, and re-adding a class the element
    // already has does nothing on its own.
    //
    // The usual restart trick is to read `offsetWidth` between the remove and
    // the add, forcing a synchronous reflow. Do not: a forced layout while a
    // smooth scroll is in flight cuts the scroll short, and that is exactly why
    // following a reference jumped instead of gliding. Two frames of waiting
    // gets the same restart without ever making the browser stop and measure.
    el.classList.remove("message-flash")
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        el.classList.add("message-flash")
        el.addEventListener(
          "animationend",
          () => el.classList.remove("message-flash"),
          { once: true }
        )
      })
    })
  }

  // Already in view, or motion is unwelcome: nothing to wait for.
  if (reduced || isFullyVisible(el)) {
    el.scrollIntoView({
      behavior: reduced ? "auto" : "smooth",
      block: "center",
    })
    sweep()
    return true
  }

  const scroller = scrollParent(el)
  let done = false
  const arrive = () => {
    if (done) return
    done = true
    scroller?.removeEventListener("scrollend", arrive)
    sweep()
  }

  // `scrollend` is the precise signal but is not everywhere yet, and it never
  // fires if the browser decides no scrolling was needed. The timer is the
  // guarantee that the sweep plays at all; whichever lands first wins.
  scroller?.addEventListener("scrollend", arrive, { once: true })
  window.setTimeout(arrive, SCROLL_SETTLE_MS)

  el.scrollIntoView({ behavior: "smooth", block: "center" })
  return true
}

// Long enough to cover a typical smooth scroll, short enough that the sweep
// still reads as a response to the click.
const SCROLL_SETTLE_MS = 600

function isFullyVisible(el: Element): boolean {
  const box = el.getBoundingClientRect()
  return box.top >= 0 && box.bottom <= window.innerHeight
}

/**
 * The element that actually scrolls this message — the thread's own overflow
 * container in chat, `<main>` elsewhere. Needed because `scrollend` does not
 * bubble up from an element scroller, so listening on the document would miss it.
 */
function scrollParent(el: HTMLElement): HTMLElement | null {
  let node = el.parentElement
  while (node) {
    const { overflowY } = getComputedStyle(node)
    if (
      (overflowY === "auto" || overflowY === "scroll") &&
      node.scrollHeight > node.clientHeight
    ) {
      return node
    }
    node = node.parentElement
  }
  return document.scrollingElement as HTMLElement | null
}

function excerpt(content: string): string {
  const plain = stripThinking(content)
    .replace(/<\|.*?\|>/g, " ")
    .replace(/\s+/g, " ")
    .trim()
  return plain.length > EXCERPT_CHARS
    ? `${plain.slice(0, EXCERPT_CHARS).trimEnd()}…`
    : plain
}
