import type { Ref } from "react"

import { FigureSlate } from "@/components/ui/figure-slate"

export interface NewChatArtProps {
  ref: Ref<HTMLDivElement>
  trailRef: Ref<HTMLDivElement>
}

export function NewChatArt({ ref, trailRef }: NewChatArtProps) {
  return (
    <div ref={ref} className="new-chat-art relative w-full" aria-hidden="true">
      <FigureSlate className="new-chat-art-rest" shimmer={false} />
      <div className="new-chat-art-lit pointer-events-none absolute inset-0">
        <FigureSlate shimmer={false} />
      </div>
      <div
        ref={trailRef}
        className="new-chat-art-trail pointer-events-none absolute inset-0"
      >
        <FigureSlate shimmer={false} />
      </div>
      <div className="new-chat-art-intro pointer-events-none absolute inset-0">
        <FigureSlate part="word" shimmer={false} />
      </div>
    </div>
  )
}
