"use client"

import Link from "next/link"
import {
  Dumbbell,
  Gauge,
  MessageSquare,
  Plus,
  Sparkles,
  ThumbsUp,
} from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

/**
 * Add job — a menu that routes to a composer, not a modal that contains one.
 *
 * Each kind's form is a full page worth of controls (the training rail alone is
 * a dozen hyperparameters), so they get their own URLs: deep-linkable, seedable
 * from a finished job's config, and survivable across a reload.
 *
 * Chat is on the list because a chat turn genuinely is a queued job now — but it
 * is composed by talking, so it points at /chat rather than a form.
 */
const OPTIONS = [
  { label: "Train a model", href: "/jobs/new/train", icon: Dumbbell },
  { label: "Benchmark a model", href: "/jobs/new/benchmark", icon: Gauge },
  { label: "Auto-label replies", href: "/jobs/new/autolabel", icon: ThumbsUp },
  { label: "Generate SFT data", href: "/jobs/new/synth", icon: Sparkles },
  { label: "Chat", href: "/chat", icon: MessageSquare },
]

export function AddJobMenu() {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={
          <Button size="sm" className="gap-1.5">
            <Plus className="size-3.5" />
            Add job
          </Button>
        }
      />
      <DropdownMenuContent align="end">
        {OPTIONS.map((option) => (
          <DropdownMenuItem
            key={option.href}
            render={<Link href={option.href} />}
          >
            <option.icon />
            {option.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
