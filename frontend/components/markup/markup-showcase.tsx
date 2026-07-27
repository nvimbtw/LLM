"use client"

import * as React from "react"

import { ChatMessageBubble } from "@/components/chat/chat-message"
import {
  MessageRefsProvider,
  buildMessageRefs,
} from "@/components/chat/message-refs"
import { Button } from "@/components/ui/button"
import type { ChatMessage } from "@/hooks/use-chat"

/**
 * Every renderable special token, in one conversation.
 *
 * One thread rather than four labelled specimens, because half of what needs
 * looking at only exists *between* messages: whether a reply reference resolves,
 * whether following it scrolls somewhere sensible, how a quote block sits under
 * the paragraph that introduced it, whether the rhythm of a real exchange holds
 * up when replies are long. A gallery of isolated fragments shows none of that.
 *
 * The rest of `SPECIAL_TOKENS` is frame rather than content — `<|user|>`,
 * `<|assistant|>`, `<|system|>` and `<|endoftext|>` delimit turns in the chat
 * template and never reach the renderer, so there is nothing to preview.
 *
 * These messages go through `ChatMessageBubble`, not a copy of it: a preview that
 * renders through its own code path stops being a preview the first time the real
 * one changes.
 *
 * Vocabulary source of truth: `SPECIAL_TOKENS` in
 * `backend/src/tokenize/src/pretokenize.rs`.
 */

interface Turn {
  id: string
  role: "user" | "assistant"
  content: string
  /** Feeds the token legend above the thread, so it cannot drift from the fixtures. */
  covers: string[]
}

const CONVERSATION: Turn[] = [
  {
    id: "m1",
    role: "user",
    content:
      "Can you explain how the tokenizer handles special tokens? A short write-up is fine.",
    covers: [],
  },
  {
    id: "m2",
    role: "assistant",
    covers: [
      "title",
      "heading",
      "subheading",
      "par",
      "bold",
      "italic",
      "code",
      "link",
      "divider",
    ],
    content:
      "<|title|>Special tokens in the tokenizer<|/title|>" +
      "<|par|>Each protected marker owns a <|bold|>single reserved id<|/bold|>, so the " +
      "assistant marker is one embedding rather than a byte triple the model has to " +
      "reassemble. That is the whole reason they are worth reserving.<|/par|>" +
      "<|heading|>Why not just use text?<|/heading|>" +
      "<|par|>Because the model cannot reliably tell <|italic|>two<|/italic|> markers apart " +
      "once they have been shredded into overlapping bytes. Every marker used to share the " +
      "prefix <|code|>0xEE 0x80<|/code|>, which is a poor thing to ask an embedding to " +
      "disambiguate.<|/par|>" +
      "<|subheading|>Layout<|/subheading|>" +
      "<|par|>The vocabulary is laid out as raw bytes, then the special block, then BPE " +
      "merges above it. See <|link|>backend/src/tokenize/src/pretokenize.rs<|/link|> for the " +
      "table itself.<|/par|>" +
      "<|divider|>" +
      "<|par|>There are 36 of them at present.<|/par|>",
  },
  {
    id: "m3",
    role: "user",
    content: "Got it. What should I check before starting a **cycle**?",
    covers: [],
  },
  {
    id: "m4",
    role: "assistant",
    covers: ["list", "item", "quote", "codeblock"],
    content:
      "<|par|>Three things, in this order:<|/par|>" +
      "<|list|>" +
      "<|item|>Some <|bold|>hand ratings<|/bold|> exist — round 1 calibrates against them " +
      "and refuses if it disagrees.<|/item|>" +
      "<|item|>A reference corpus is installed, or the judge has no English to score " +
      "against.<|/item|>" +
      "<|item|>The base model has a tokenizer. Run <|code|>slate models inspect<|/code|> if " +
      "you are unsure.<|/item|>" +
      "<|/list|>" +
      "<|quote|>A cycle started without ratings halts immediately, on a fact that was " +
      "knowable before it started.<|/quote|>" +
      "<|par|>The config it freezes looks like this:<|/par|>" +
      "<|codeblock|>{\n" +
      '  "max_rounds": 5,\n' +
      '  "raft_lr": 1e-5,\n' +
      '  "kto_lr": 5e-6,\n' +
      '  "beta": 0.1,\n' +
      '  "min_gain": 0.002\n' +
      "}<|/codeblock|>",
  },
  {
    id: "m5",
    role: "user",
    content:
      "I set `min_gain` to 0 and the cycle ran all night without stopping. Is that expected?",
    covers: [],
  },
  {
    id: "m6",
    role: "assistant",
    covers: ["reply", "warn"],
    content:
      "<|reply|>m5<|/reply|>" +
      "<|par|>Expected, yes — and worth guarding against.<|/par|>" +
      "<|warn|>A <|code|>min_gain<|/code|> of zero means any score above the previous best " +
      "counts as an improvement, including noise. The loop will keep going until it hits " +
      "<|code|>max_rounds<|/code|>, and the model it keeps may be no better than the one it " +
      "started from.<|/warn|>" +
      "<|par|>Set it back to <|code|>0.002<|/code|> — the default in the config above — and " +
      "the cycle halts as soon as a round stops paying for itself.<|/par|>",
  },
  {
    id: "m7",
    role: "user",
    content: "Is 8,191 prime?",
    covers: [],
  },
  {
    id: "m8",
    role: "assistant",
    covers: ["thinking"],
    content:
      // Over the collapse threshold, and with a blank line in it, so this turn
      // covers both the folding and the paragraph split.
      "<|thinking|>8191 = 2^13 - 1. Mersenne numbers have the form 2^p - 1, and a necessary " +
      "condition for one to be prime is that the exponent p is itself prime. 13 is prime, so " +
      "8191 is at least a candidate — that alone does not settle it, since 2^11 - 1 = 2047 = " +
      "23 × 89 is composite despite 11 being prime.\n\n" +
      "Checking directly: to rule out composites I only need trial divisors up to " +
      "sqrt(8191) ≈ 90.5. Any factor of a Mersenne number 2^p - 1 must be congruent to 1 mod " +
      "2p, so candidates are 27, 53 and 79 — of which only 53 and 79 are prime, and neither " +
      "divides 8191. It is prime, and it is in fact the fourth Mersenne prime.<|/thinking|>" +
      "<|par|>Yes — and it is a notable one.<|/par|>" +
      "<|par|>8,191 is <|bold|>2^13 − 1<|/bold|>, the fourth Mersenne prime.<|/par|>" +
      "<|thinking|>Should I list the earlier ones? It is short and makes the pattern " +
      "concrete. Yes.<|/thinking|>" +
      "<|par|>The ones before it are 3, 7 and 31 — that is 2^2−1, 2^3−1 and 2^5−1.<|/par|>",
  },
  {
    id: "m9",
    role: "user",
    content: "One more: can you point back at something you said earlier?",
    covers: [],
  },
  {
    id: "m10",
    role: "assistant",
    covers: ["reply"],
    content:
      "<|reply|>m4<|/reply|>" +
      "<|par|>That one was mine rather than yours, so the header says so. Clicking it " +
      "scrolls up to the message and flashes it — which is the part worth trying, since a " +
      "reference you cannot follow is only half a reference.<|/par|>" +
      "<|reply|>m-deleted<|/reply|>" +
      "<|par|>And this one points at a message that is not in the thread. It still renders: " +
      "a reference that vanished silently would leave the paragraph under it reading as a " +
      "non-sequitur.<|/par|>",
  },
]

// Turns carry `covers`; the legend is derived so it cannot fall out of step with
// what the fixtures actually contain.
const COVERED = Array.from(new Set(CONVERSATION.flatMap((t) => t.covers)))

// Rendered but not emitted by any model yet.
const UNSHIPPED = new Set(["quote", "warn", "reply"])

export function MarkupShowcase() {
  const [showSource, setShowSource] = React.useState(false)
  const lookup = React.useMemo(() => buildMessageRefs(CONVERSATION), [])

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <ul className="flex flex-wrap items-center gap-1.5">
          {COVERED.map((token) => (
            <li
              key={token}
              className={
                UNSHIPPED.has(token)
                  ? "rounded-sm bg-kind-gather/14 px-1.5 py-0.5 font-mono text-xs text-kind-gather"
                  : "rounded-sm bg-muted px-1.5 py-0.5 font-mono text-xs text-muted-foreground"
              }
              title={
                UNSHIPPED.has(token)
                  ? "Rendered here, not emitted by any model yet"
                  : undefined
              }
            >
              {token}
            </li>
          ))}
        </ul>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowSource((v) => !v)}
        >
          {showSource ? "Hide markup" : "Show markup"}
        </Button>
      </div>

      {/* Scoped to this thread, exactly as a real conversation scopes its own —
          a reference can only point at a message in the thread it lives in. */}
      <MessageRefsProvider lookup={lookup}>
        <div className="rounded-lg border border-border bg-card px-5 py-8 sm:px-8">
          {/* gap-7 matches the real thread, so the rhythm previewed here is the
              rhythm you get in a conversation. */}
          <div className="flex flex-col gap-7">
            {CONVERSATION.map((turn) => (
              <div key={turn.id} className="flex flex-col gap-2">
                <ChatMessageBubble
                  message={
                    {
                      // No byline: these were never persisted, and a fixture
                      // claiming a model and a token count would be a lie.
                      id: turn.id,
                      role: turn.role,
                      content: turn.content,
                    } as ChatMessage
                  }
                />
                {showSource && (
                  <pre className="overflow-x-auto rounded-md border border-border bg-muted px-3 py-2">
                    <code className="font-mono text-xs break-all whitespace-pre-wrap text-muted-foreground">
                      {turn.content}
                    </code>
                  </pre>
                )}
              </div>
            ))}
          </div>
        </div>
      </MessageRefsProvider>
    </div>
  )
}
