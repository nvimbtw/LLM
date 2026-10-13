"use client"

import dynamic from "next/dynamic"
import { useSettings } from "@/components/settings/settings-provider"
import {
  TrellisThrobber,
  type ThrobberProps as TrellisProps,
} from "./trellis-throbber"
import type { ArtStream } from "@/lib/live-throbber/stream"

export type { ThrobberMode } from "./trellis-throbber"

export interface ThrobberProps extends TrellisProps {
  stream?: ArtStream
  /** Fixture override; otherwise follow the account's preference. */
  live?: boolean
}

const LiveArt = dynamic(
  () =>
    import("@/components/throbber/live-art").then((module) => module.LiveArt),
  {
    ssr: false,
    loading: () => (
      <span className="inline-flex size-(--size-live-throbber) shrink-0 items-center justify-center">
        <TrellisThrobber />
      </span>
    ),
  }
)

export function Throbber({ live, stream, ...props }: ThrobberProps) {
  const { settings } = useSettings()
  return (live ?? settings.useLiveThrobber) ? (
    <LiveArt {...props} stream={stream} />
  ) : (
    <TrellisThrobber {...props} />
  )
}
