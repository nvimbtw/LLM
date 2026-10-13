"use client"

import "./new-chat-scene.css"

import { useEffect, useRef, useState, type ReactNode } from "react"

import { SLATE_GRID } from "@/components/ui/figure-slate"
import { usePointerLight } from "@/hooks/use-pointer-light"

import { SlateTexture } from "./slate-texture"
import { NewChatArt } from "./new-chat-art"

const TRAIL_LIFETIME_MS = 1800
const TRAIL_RADIUS_REM = 7

export interface NewChatSceneProps {
  children: ReactNode
}

interface GridPosition {
  x: number
  y: number
  scale: number
}

export function NewChatScene({ children }: NewChatSceneProps) {
  const artRef = useRef<HTMLDivElement>(null)
  const { scene, trail, illuminate, dim } = usePointerLight({
    trailLifetimeMs: TRAIL_LIFETIME_MS,
    trailRadiusRem: TRAIL_RADIUS_REM,
  })
  const [grid, setGrid] = useState<GridPosition | null>(null)

  useEffect(() => {
    const element = scene.current
    const art = artRef.current
    if (!element || !art) return
    const observer = new ResizeObserver(() => {
      const bounds = element.getBoundingClientRect()
      const figure = art.getBoundingClientRect()
      const position = {
        x: figure.left - bounds.left,
        y: figure.top - bounds.top,
        scale: figure.width / SLATE_GRID.width,
      }
      element.style.setProperty("--slate-art-x", `${position.x}px`)
      element.style.setProperty("--slate-art-y", `${position.y}px`)
      setGrid((previous) =>
        previous?.x === position.x &&
        previous.y === position.y &&
        previous.scale === position.scale
          ? previous
          : position
      )
    })
    observer.observe(element)
    observer.observe(art)
    // Composer height also moves the centered artwork without resizing it.
    if (art.parentElement?.parentElement)
      observer.observe(art.parentElement.parentElement)
    return () => observer.disconnect()
  }, [scene, artRef])

  return (
    <section
      ref={scene}
      onPointerMove={illuminate}
      onPointerLeave={dim}
      className="new-chat-scene relative isolate flex flex-1 flex-col overflow-hidden"
    >
      {grid && (
        <div
          className="pointer-events-none absolute inset-0 -z-10"
          aria-hidden="true"
        >
          <div className="new-chat-grid-rest absolute inset-0">
            <SlateTexture {...grid} />
          </div>
        </div>
      )}
      <div className="mx-auto flex w-full max-w-2xl flex-1 flex-col items-center justify-center px-4">
        <h1 className="flex w-full items-end justify-center">
          <span className="sr-only">Slate</span>
          <NewChatArt ref={artRef} trailRef={trail} />
        </h1>
        <div className="mt-5 w-full">{children}</div>
      </div>
    </section>
  )
}
