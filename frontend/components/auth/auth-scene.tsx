"use client"

import "./auth-scene.css"

import type { ReactNode } from "react"

import { usePointerLight } from "@/hooks/use-pointer-light"

import { AuthTexture } from "./auth-texture"

export interface AuthSceneProps {
  children: ReactNode
}

export function AuthScene({ children }: AuthSceneProps) {
  const { scene, trail, illuminate, dim } = usePointerLight()

  return (
    <main
      ref={scene}
      onPointerMove={illuminate}
      onPointerLeave={dim}
      className="relative isolate flex min-h-svh items-center justify-center overflow-hidden px-6 py-16 sm:px-10 sm:py-20"
    >
      <div
        className="auth-art pointer-events-none absolute inset-0 -z-10"
        aria-hidden="true"
      >
        <div className="auth-art-rest absolute inset-0">
          <AuthTexture />
        </div>
        <div className="auth-art-lit absolute inset-0">
          <AuthTexture />
        </div>
        <div
          ref={trail}
          className="auth-art-lit auth-art-trail absolute inset-0"
        >
          <AuthTexture />
        </div>
        <div className="auth-art-lit auth-art-core absolute inset-0">
          <AuthTexture />
        </div>
      </div>
      {children}
    </main>
  )
}
